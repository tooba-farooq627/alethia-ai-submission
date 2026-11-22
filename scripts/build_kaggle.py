"""Prepare a private Kaggle migration without changing inference."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

from build_colab import code, markdown

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel', action='store_true', help='Use both T4 GPUs with disjoint shard workers')
    parser.add_argument('--background', action='store_true', help='Use an authorized Kaggle Secret for Save & Run All')
    parser.add_argument('--secret-name', default='alethia-finevideo-read', help='Secret label only; never pass the token')
    parser.add_argument('--bundle-input', type=Path, help='Write the verified bundle separately for attachment as a private Kaggle input')
    args = parser.parse_args()
    if args.background and not args.parallel:
        parser.error('Background runs require the bounded parallel runner')
    manifest = json.loads((args.checkpoint / 'manifest.json').read_text())
    sources = sorted((ROOT / 'video_indexer').glob('*.py')) + [
        ROOT / 'scripts/resumable_run.py', ROOT / 'requirements.txt', ROOT / 'models/yolo11l.pt']
    signature = hashlib.sha256(json.dumps({
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'revision': manifest['dataset_revision'], 'count': 50,
        'batch_size': 8, 'precision': 'fp16', 'decode_threads': 2,
    }, sort_keys=True).encode()).hexdigest()
    if signature != manifest['signature']:
        raise ValueError('Source/model/config differs from the saved run')
    checkpoint_files = [args.checkpoint / 'manifest.json']
    for video, metrics in manifest['videos'].items():
        shard, row = video.rsplit(':', 1)
        part = args.checkpoint / 'videos' / shard / f'{int(row):05d}.jsonl.gz'
        metadata = part.with_suffix('.metrics.json')
        if digest(part) != metrics['sha256'] or json.loads(metadata.read_text()) != metrics:
            raise ValueError('Checkpoint failed verification: ' + video)
        checkpoint_files += [part, metadata]
    for shard, metrics in manifest['shards'].items():
        part = args.checkpoint / (shard + '.jsonl.gz')
        if digest(part) != metrics['sha256']:
            raise ValueError('Shard failed verification: ' + shard)
        checkpoint_files.append(part)
    # Explicit allowlist: no credentials, chats, original videos, or unrelated files.
    project_files = [ROOT / name for name in ('run.py', 'prepare.py', 'requirements.txt', 'requirements-dev.txt', 'pytest.ini')]
    project_files += sorted((ROOT / 'video_indexer').glob('*.py'))
    project_files += sorted((ROOT / 'tests').glob('*.py'))
    project_files += [p for p in sorted((ROOT / 'scripts').glob('*.py')) if not p.name.startswith('build_')]
    stream = io.BytesIO()
    # Compress the entire bundle so repeated per-video metadata compresses together.
    # Keeping the serialized notebook below 1 MiB also avoids large editor imports.
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_STORED) as bundle:
        for path in project_files:
            bundle.writestr('source/' + str(path.relative_to(ROOT)), path.read_bytes())
        for path in checkpoint_files:
            bundle.writestr('checkpoint/' + str(path.relative_to(args.checkpoint)), path.read_bytes())
    packed = zlib.compress(stream.getvalue(), 9)
    if args.bundle_input:
        args.bundle_input.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_input.write_bytes(packed)
        restore = f'''bundles = list(Path('/kaggle/input').rglob({args.bundle_input.name!r}))
assert len(bundles) == 1, 'Attach the private verified checkpoint bundle'
assert hashlib.sha256(bundles[0].read_bytes()).hexdigest() == {hashlib.sha256(packed).hexdigest()!r}, 'Input bundle hash differs'
bundle_bytes = zlib.decompress(bundles[0].read_bytes())'''
    else:
        payload = base64.b64encode(packed).decode()
        restore = f'payload = {payload!r}\nbundle_bytes = zlib.decompress(base64.b64decode(payload))'
    runner_command = ("'scripts/parallel_run.py', '--devices', '0,1', '--max-hours', '10'"
                      if args.parallel else "'scripts/resumable_run.py'")
    run_extra = '' if args.parallel else ", '--count', '50'"
    parallel_description = ('This notebook uses both GPUs, assigning disjoint shards to separate processes. '
                            'Only the coordinator writes the manifest. It pauses after about 10 hours '
                            'at a video boundary so checkpoints can be downloaded before the session limit. '
                            'A new session must be resumed to finish any remaining shards.'
                            if args.parallel else 'This notebook uses only GPU 0; it does not yet exploit a second GPU.')
    cells = [markdown('''# Alethia AI: resume verified checkpoints on Kaggle

Keep this notebook private. Check your actual GPU quota and enable an available NVIDIA GPU and Internet in notebook settings. It makes no purchase.

This interactive notebook embeds the verified completed results and source, with no credentials or original dataset videos. It retains FP16, batch 8, every frame, the pinned dataset revision, and the original checkpoint signature. Setup must pass before resuming.

Kaggle sessions have a time limit. `/tmp` is disposable; completed output is stored under `/kaggle/working`. Download the recovery ZIP before ending a session. Verify the platform's persistence settings rather than assuming that an interactive session is a durable backup. A hidden authentication prompt requires an interactive session; do not use Save & Run All with it unattended.
''' + '\n' + parallel_description + '\n'), code(f'''import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile, zlib
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'
project = Path('/tmp/alethia')
checkpoint = Path('/kaggle/working/alethia-checkpoints')
project.mkdir(parents=True, exist_ok=True)
checkpoint.mkdir(parents=True, exist_ok=True)
restore_checkpoint = not (checkpoint / 'manifest.json').exists()
{restore}
with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:
    # Commit the manifest last so an interrupted first restore can be repeated.
    for name in sorted(bundle.namelist(), key=lambda name: name == 'checkpoint/manifest.json'):
        prefix, relative = name.split('/', 1)
        destination = (project if prefix == 'source' else checkpoint) / relative
        if prefix == 'checkpoint' and not restore_checkpoint:
            continue  # Never overwrite a newer runtime checkpoint with embedded old data.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bundle.read(name))
os.chdir(project)
print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True))
print('Checkpoint videos:', len(json.loads((checkpoint / 'manifest.json').read_text())['videos']))
'''), code('''def checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    print((result.stdout + result.stderr)[-4000:])
    result.check_returncode()

checked([sys.executable, '-m', 'pip', 'install', 'uv==0.9.5'])
python = str(project / '.venv/bin/python')
if not Path(python).exists():
    checked([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '.venv'])
checked([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python, '-r', 'requirements.txt', '-r', 'requirements-dev.txt'])
if not shutil.which('ffprobe') or subprocess.run([python, '-c', 'import cv2'], capture_output=True).returncode:
    checked(['apt-get', 'update', '-qq'])
    checked(['apt-get', 'install', '-y', '-qq', 'ffmpeg', 'libgl1', 'libglib2.0-0'])
checked([python, '-m', 'pytest', '-q'])
checked([python, 'prepare.py'])
checked([python, 'scripts/gpu_check.py'])
'''), code('''# Enter only after authorizing use of the limited FineVideo token on Kaggle.
# It stays in runtime memory, not notebook source, outputs, or a login file.
from getpass import getpass
token = getpass('FineVideo read token (hidden): ')
os.environ['HF_TOKEN'] = token
del token
checked([python, '-c', 'from huggingface_hub import HfApi; HfApi().whoami(); print("Authentication verified")'])
'''), code(f'''revision = {manifest['dataset_revision']!r}
log_path = Path('/kaggle/working/alethia-run.log')
# Refuse duplicate workers in the same live kernel.
assert 'worker' not in globals() or worker.poll() is not None, 'Worker already running'
log_handle = log_path.open('a')
worker = subprocess.Popen([python, '-u', {runner_command}, '--checkpoints', str(checkpoint), '--revision', revision{run_extra}], stdout=log_handle, stderr=subprocess.STDOUT)
print('Worker PID:', worker.pid)
'''), code('''# Monitor progress and refresh a verified recovery ZIP every ten minutes.
sys.path.insert(0, str(project))
from scripts.watch_run import monitor
monitor(worker, project, checkpoint, log_path)
''')]
    if args.parallel:
        cells.insert(4, code(f'''# Check both GPUs on three complete videos before starting the full run.
subprocess.run([python, '-u', 'scripts/validate_migration.py', '--checkpoints', str(checkpoint), '--revision', {manifest['dataset_revision']!r}, '--devices', '0,1'], check=True)
'''))
    if args.background:
        cells[0] = markdown('''# Alethia AI: private background recovery run

Use **Save Version > Save & Run All** with T4 x2 and Internet enabled. A saved background version continues independently of an interactive browser session, subject to Kaggle's runtime and quota limits. The bounded runner pauses after ten hours at a completed-video boundary; its final recovery ZIP must be attached/embedded for the next chunk. A completed Kaggle version can therefore represent a checkpoint pause, not a completed 50-shard dataset.

This notebook contains code and verified result checkpoints, with no token value or original videos embedded. Its authentication cell retrieves the separately authorized, private Kaggle Secret. Keep the notebook and its outputs private. It runs setup checks and the both-GPU all-frame comparison before launching the full runner, then monitors until the worker exits and creates the final verified recovery ZIP. Inspect the manifest's full-50 audit rather than assuming a green version status means the dataset is finished.
''')
        cells[3] = code(f'''# Authorize only this notebook to use the private, limited dataset Secret.
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret({args.secret_name!r})
checked([python, '-c', 'from huggingface_hub import HfApi; HfApi().whoami(); print("Authentication verified")'])
''')
        cells.append(code('''# Fail the saved version on a failed worker; a clean checkpoint pause is allowed.
assert worker.returncode == 0, 'Background worker failed; inspect alethia-run.log and recovery ZIP'
print('BACKGROUND_CHUNK_FINISHED: inspect full_50_shards before calling the assignment complete')
'''))
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                             'language_info': {'name': 'python'}}, 'cells': cells}
    for index, cell in enumerate(cells):
        cell['id'] = f'alethia-kaggle-{index:02d}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Prepared {args.output}; verified {len(manifest["videos"])} videos; not executed on Kaggle')


if __name__ == '__main__':
    main()
