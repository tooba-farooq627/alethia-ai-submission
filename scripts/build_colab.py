"""Build a portable setup notebook with only this project's source embedded."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
            "outputs": [], "execution_count": None}


def main():
    stream = io.BytesIO()
    files = ([ROOT / p for p in ("run.py", "prepare.py", "requirements.txt", "requirements-dev.txt", "pytest.ini", "README.md", "ITERATIONS.md")]
             + sorted((ROOT / "video_indexer").glob("*.py"))
             + sorted((ROOT / "tests").glob("*.py"))
             + [p for p in sorted((ROOT / "scripts").glob("*.py")) if p.name != "build_colab.py"])
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
    payload = base64.b64encode(stream.getvalue()).decode()
    cells = [markdown("""# Alethia AI - setup and all-frame baseline

Use the personal account approved for this task. Select **Runtime > Change runtime type > T4 GPU** (or an available NVIDIA GPU). This notebook does not purchase compute.

Run setup and tests first. Setup creates a separate Python 3.11 environment even when the notebook kernel is newer. Real data requires your own FineVideo terms acceptance and a read token entered privately. Study/implementation iterations are described in `ITERATIONS.md`. The assignment write-up must be written without AI.
"""), code("""import platform, shutil, subprocess
from pathlib import Path
print('Python:', platform.python_version())
print('Free disk GiB:', round(shutil.disk_usage('/content').free / 2**30, 1))
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
print(gpu.stdout or gpu.stderr)
assert gpu.returncode == 0, 'Select an NVIDIA GPU runtime before running the dataset.'
"""), code(f"""import base64, io, os, zipfile
project = Path('/content/alethia')
project.mkdir(exist_ok=True)
payload = {payload!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
    bundle.extractall(project)
os.chdir(project)
print('Project source ready:', project)
"""), code("""import sys
os.environ['MPLBACKEND'] = 'Agg'
def checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    print((result.stdout + result.stderr)[-6000:])
    result.check_returncode()

checked([sys.executable, '-m', 'pip', 'install', 'uv==0.9.5'])
python = str(project / '.venv/bin/python')
if not Path(python).exists():
    checked([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '.venv'])
checked([sys.executable, '-m', 'uv', 'pip', 'install', '--python', python, '-r', 'requirements.txt', '-r', 'requirements-dev.txt'])
opencv_ready = subprocess.run([python, '-c', 'import cv2'], capture_output=True).returncode == 0
if not shutil.which('ffprobe') or not opencv_ready:
    checked(['apt-get', 'update', '-qq'])
    checked(['apt-get', 'install', '-y', '-qq', 'ffmpeg', 'libgl1', 'libglib2.0-0'])
checked([python, '-m', 'pytest', '-q'])
checked([python, 'prepare.py'])
checked([python, 'scripts/capture_hardware.py', '--output', 'artifacts/hardware.json'])
checked([python, 'scripts/gpu_check.py'])
"""), markdown("""## Private authentication and first shard
Accept the dataset terms yourself: https://huggingface.co/datasets/HuggingFaceFV/finevideo

The next cell requests a token only when no valid login exists. A new token stays in the current runtime's environment and is not written into this notebook or a login file. The hidden prompt does not print it. Do not paste a token into source, output, or AI chats.
"""), code("""from getpass import getpass
from huggingface_hub import HfApi
try:
    HfApi().whoami()
    print('Existing Hugging Face login available.')
except Exception:
    token = getpass('Hugging Face read token (hidden): ')
    HfApi(token=token).whoami()
    os.environ['HF_TOKEN'] = token
    del token
dataset_revision = HfApi().dataset_info('HuggingFaceFV/finevideo').sha
print('Pinned dataset revision:', dataset_revision)
subprocess.run([python, 'scripts/download_data.py', '--count', '1', '--revision', dataset_revision], check=True)
subprocess.run([python, 'scripts/make_subset.py', '--source', 'finevideo/data/train-00000-of-01357.parquet', '--output', 'sample', '--rows', '3'], check=True)
"""), markdown("""## Real-data baseline and batch validation
These runs use the same three videos and every frame. Keep both outputs and timing reports. The full 50-shard run remains a separate step.
"""), code("""subprocess.run([python, 'run.py', '--input', 'sample', '--output', 'artifacts/reference.jsonl', '--batch-size', '1', '--precision', 'fp32', '--profile', 'artifacts/reference-profile.json'], check=True)
subprocess.run([python, 'run.py', '--input', 'sample', '--output', 'artifacts/batched.jsonl', '--batch-size', '8', '--precision', 'auto', '--profile', 'artifacts/batched-profile.json'], check=True)
subprocess.run([python, 'scripts/compare_results.py', '--reference', 'artifacts/reference.jsonl', '--candidate', 'artifacts/batched.jsonl'], check=True)
"""), markdown("""## Full run with durable checkpoints
Choose persistent storage before starting. Colab `/content` is temporary. If you decide to mount Google Drive, do so yourself using Colab's normal authorization flow; that grants separate Drive access. Set `checkpoint_dir` to a directory on that storage. The default deliberately fails until you choose it.

The recovery runner uses the validated CUDA FP16/batch-8 configuration. Use the same pinned dataset revision and identical source/configuration when resuming. Each completed video has an atomic, hash-verified checkpoint; an interrupted video is recomputed. Downloads occur one shard at a time. Historical first/full timings must come from actual completed runs; subset estimates are not substitutes. Use `run.py` for the evaluator-equivalent offline processing measurement.
"""), code("""checkpoint_dir = None  # Set a persistent path, e.g. an explicitly authorized mounted Drive folder.
assert checkpoint_dir is not None, 'Choose persistent checkpoint storage first.'
from huggingface_hub import HfApi
if 'dataset_revision' not in globals():
    dataset_revision = HfApi().dataset_info('HuggingFaceFV/finevideo').sha
print('Pinned dataset revision:', dataset_revision)
subprocess.run([python, 'scripts/resumable_run.py', '--checkpoints', str(checkpoint_dir), '--count', '50', '--revision', dataset_revision], check=True)
"""), markdown("""## Preserve evidence
Save hardware metadata, manifests, first/final outputs, and complete AI chats. A subset run is not a completed full run. No full-dataset runtime or accuracy is claimed until measured. Write `WRITEUP.md` independently.
""")]
    for index, cell in enumerate(cells):
        cell['id'] = f'alethia-{index:02d}'
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "Alethia_AI_Setup_and_Baseline.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}
    output = ROOT / "Colab_Setup.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Created {output} with {len(cells)} cells; embedded {len(files)} source files")


if __name__ == "__main__":
    main()
