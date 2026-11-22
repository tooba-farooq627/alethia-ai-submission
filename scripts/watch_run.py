"""Notebook monitor and portable snapshots of committed results only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def build_snapshot(project, checkpoint, state, destination):
    """Snapshot immutable, committed members; omit active videos and credentials."""
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('manifest.json', json.dumps(state, indent=2) + '\n')
        for video, metrics in state['videos'].items():
            shard, row = video.rsplit(':', 1)
            part = checkpoint / 'videos' / shard / f'{int(row):05d}.jsonl.gz'
            if hashlib.sha256(part.read_bytes()).hexdigest() != metrics['sha256']:
                raise ValueError('Snapshot member hash differs: ' + video)
            if json.loads(part.with_suffix('.metrics.json').read_text()) != metrics:
                raise ValueError('Snapshot member metadata differs: ' + video)
            for path in (part, part.with_suffix('.metrics.json')):
                bundle.write(path, str(path.relative_to(checkpoint)))
        for shard, metrics in state['shards'].items():
            part = checkpoint / (shard + '.jsonl.gz')
            if hashlib.sha256(part.read_bytes()).hexdigest() != metrics['sha256']:
                raise ValueError('Snapshot shard hash differs: ' + shard)
            bundle.write(part, part.name)
        if state.get('full_50_shards'):
            final = checkpoint / 'results.jsonl.gz'
            if hashlib.sha256(final.read_bytes()).hexdigest() != state['results_sha256']:
                raise ValueError('Snapshot result hash differs')
            bundle.write(final, 'results.jsonl.gz')
        # Explicit diagnostics/source allowlist: no environment, token files, or videos.
        diagnostics = list((checkpoint / 'migration-validation').rglob('*'))
        diagnostics += list(checkpoint.glob('gpu-*.log'))
        for path in diagnostics:
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.gz', '.log'}:
                bundle.write(path, str(path.relative_to(checkpoint)))
        files = [project / name for name in ('run.py', 'prepare.py', 'requirements.txt', 'requirements-dev.txt', 'pytest.ini')]
        files += sorted((project / 'video_indexer').glob('*.py'))
        files += sorted((project / 'scripts').glob('*.py'))
        for path in files:
            bundle.write(path, 'implementation/' + str(path.relative_to(project)))
    temporary.replace(destination)


def monitor(worker, project, checkpoint, log_path):
    from datetime import datetime, timezone
    from IPython.display import FileLink, clear_output, display
    project, checkpoint, log_path = map(Path, (project, checkpoint, log_path))
    snapshot = checkpoint.parent / 'alethia-recovery.zip'
    last_snapshot = None
    while True:
        state = json.loads((checkpoint / 'manifest.json').read_text())
        exit_code = worker.poll()
        complete = bool(state.get('full_50_shards') and state.get('audit', {}).get('valid'))
        status = {'utc': datetime.now(timezone.utc).isoformat(),
                  'worker_pid': worker.pid, 'worker_exit': exit_code,
                  'state': 'complete_audited' if complete else ('running' if exit_code is None else
                           'paused_with_checkpoints' if exit_code == 0 else 'failed'),
                  'videos': len(state['videos']), 'shards': len(state['shards']),
                  'seconds': sum(v.get('seconds', 0) for v in state['videos'].values()),
                  'errors': sum(v.get('errors', 0) for v in state['videos'].values()),
                  'last_checkpoint_utc': state.get('updated_utc')}
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader'], capture_output=True, text=True)
        status['gpu_utilization_and_mib'] = gpu.stdout.strip().splitlines()
        if last_snapshot is None or time.monotonic() - last_snapshot >= 600 or exit_code is not None:
            build_snapshot(project, checkpoint, state, snapshot)
            last_snapshot = time.monotonic()
        status['snapshot_bytes'] = snapshot.stat().st_size
        status_path = checkpoint.parent / 'alethia-status.json'
        status_path.with_suffix('.tmp').write_text(json.dumps(status, indent=2) + '\n')
        status_path.with_suffix('.tmp').replace(status_path)
        clear_output(wait=True)
        print(json.dumps(status, separators=(',', ':')), flush=True)
        if log_path.exists():
            print(log_path.read_text()[-1000:])
        display(FileLink(str(snapshot)))
        display(FileLink(str(status_path)))
        if exit_code is not None:
            break
        time.sleep(30)
