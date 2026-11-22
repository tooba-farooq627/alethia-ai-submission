"""Gate a new GPU environment against three verified all-frame video results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_results import compare
from scripts.full_run import digest, now
from scripts.parallel_run import legacy_signature, verify_existing
from scripts.resumable_run import atomic_json


def gate(result):
    return bool(result['reference_seconds'] and result['object_micro_f1'] >= .995
                and result['people_exact'] >= .99 and result['people_within_one'] >= .999
                and not result['missing_seconds'] and not result['extra_seconds']
                and not result['reference_errors'] and not result['candidate_errors'])


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    manifest = json.loads((args.checkpoints / 'manifest.json').read_text())
    if manifest['signature'] != legacy_signature(args.revision):
        raise ValueError('Source/model/config differs from original checkpoint signature')
    verify_existing(args.checkpoints, manifest)
    name = 'train-00000-of-01357.parquet'
    reference_parts = [args.checkpoints / 'videos' / name / f'{row:05d}.jsonl.gz' for row in range(3)]
    if any(f'{name}:{row}' not in manifest['videos'] for row in range(3)):
        raise ValueError('Need the original first three complete video checkpoints')
    devices = args.devices.split(',')
    if len(set(devices)) != len(devices) or not all(device.isdecimal() for device in devices):
        raise ValueError('Devices must be distinct numeric GPU indices')
    directory = args.checkpoints / 'migration-validation' / now().replace(':', '-').replace('+', '_')
    directory.mkdir(parents=True)
    reference = directory / 'reference.jsonl.gz'
    with reference.open('wb') as output:
        for path in reference_parts:
            with path.open('rb') as member:
                shutil.copyfileobj(member, output)
    source = Path(hf_hub_download('HuggingFaceFV/finevideo', repo_type='dataset',
                  filename='data/' + name, revision=args.revision, local_dir=ROOT / 'finevideo'))
    sample = ROOT / ('migration-sample-' + directory.name)
    subprocess.run([sys.executable, str(ROOT / 'scripts/make_subset.py'), '--source', str(source),
                    '--output', str(sample), '--rows', '3'], check=True)
    processes, logs = [], []
    started = time.perf_counter()
    try:
        for device in devices:
            log = (directory / f'gpu-{device}.log').open('w')
            logs.append(log)
            command = [sys.executable, '-u', str(ROOT / 'run.py'), '--input', str(sample),
                       '--output', str(directory / f'gpu-{device}.jsonl'), '--device', '0',
                       '--batch-size', '8', '--precision', 'fp16', '--decode-threads', '2',
                       '--profile', str(directory / f'gpu-{device}-profile.json')]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': device, 'MPLBACKEND': 'Agg'}))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('Migration validation worker failed; inspect gpu logs')
            print(json.dumps({'phase': 'validating_three_complete_videos_on_each_gpu',
                              'elapsed_seconds': round(time.perf_counter() - started),
                              'worker_exits': [process.poll() for process in processes]}), flush=True)
            time.sleep(30)
        if any(process.returncode for process in processes):
            raise RuntimeError('Migration validation worker failed; inspect gpu logs')
        results = {device: compare(reference, directory / f'gpu-{device}.jsonl') for device in devices}
        report = {'completed_utc': now(), 'reference_sha256': digest(reference),
                  'legacy_signature': manifest['signature'],
                  'wall_seconds': time.perf_counter() - started, 'devices': results,
                  'passed': all(gate(result) for result in results.values())}
        atomic_json(directory / 'report.json', report)
        print(json.dumps(report), flush=True)
        if not report['passed']:
            raise RuntimeError('Migration output agreement gate failed; full run not started')
        print('MIGRATION_VALIDATION_PASSED', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
        shutil.rmtree(sample)


if __name__ == '__main__':
    main()
