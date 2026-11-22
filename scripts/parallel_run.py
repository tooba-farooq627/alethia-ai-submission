"""Partition complete shards across GPUs, preserving the validated v2 algorithm."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.full_run import digest, now
from scripts.resumable_run import atomic_json, save_video


def legacy_signature(revision):
    files = sorted((ROOT / 'video_indexer').glob('*.py')) + [
        ROOT / 'scripts/resumable_run.py', ROOT / 'requirements.txt', ROOT / 'models/yolo11l.pt']
    return hashlib.sha256(json.dumps({
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'revision': revision, 'count': 50, 'batch_size': 8,
        'precision': 'fp16', 'decode_threads': 2,
    }, sort_keys=True).encode()).hexdigest()


def read_part(part):
    metrics = json.loads(part.with_suffix('.metrics.json').read_text())
    if metrics.get('videos') != 1 or digest(part) != metrics['sha256']:
        raise ValueError('Invalid video checkpoint: ' + str(part))
    return metrics


def collect(folder, manifest):
    """Only the coordinator writes the manifest; workers own disjoint shard paths."""
    changed = False
    for metadata in sorted((folder / 'videos').glob('train-*-of-01357.parquet/*.jsonl.metrics.json')):
        shard = metadata.parent.name
        match = re.fullmatch(r'train-(\d{5})-of-01357\.parquet', shard)
        if not match or not 0 <= int(match[1]) < 50:
            raise ValueError('Unexpected checkpoint shard: ' + shard)
        row_text = metadata.name.removesuffix('.jsonl.metrics.json')
        if not re.fullmatch(r'\d{5}', row_text):
            raise ValueError('Unexpected checkpoint row: ' + str(metadata))
        video = f'{shard}:{int(row_text)}'
        metrics = json.loads(metadata.read_text())
        if manifest['videos'].get(video) == metrics:
            continue
        part = metadata.with_name(row_text + '.jsonl.gz')
        metrics = read_part(part)
        if video in manifest['videos']:
            raise ValueError('Existing committed video metadata changed: ' + video)
        manifest['videos'][video] = metrics
        changed = True
    for metadata in sorted(folder.glob('train-*-of-01357.parquet.shard-metrics.json')):
        shard = metadata.name.removesuffix('.shard-metrics.json')
        if shard not in {f'train-{index:05d}-of-01357.parquet' for index in range(50)}:
            raise ValueError('Unexpected checkpoint shard: ' + shard)
        metrics = json.loads(metadata.read_text())
        if manifest['shards'].get(shard) == metrics:
            continue
        if shard in manifest['shards']:
            raise ValueError('Existing committed shard metadata changed: ' + shard)
        expected = metrics['expected_videos']
        if type(expected) is not int or expected < 1:
            raise ValueError('Invalid expected video count')
        members = [manifest['videos'][f'{shard}:{row}'] for row in range(expected)]
        for key in ('videos', 'seconds', 'errors', 'frames', 'wall_seconds'):
            if metrics.get(key, 0) != sum(m.get(key, 0) for m in members):
                raise ValueError('Shard totals disagree: ' + shard)
        if digest(folder / (shard + '.jsonl.gz')) != metrics['sha256']:
            raise ValueError('Shard checkpoint changed: ' + shard)
        manifest['shards'][shard] = metrics
        changed = True
    if changed:
        manifest['updated_utc'] = now()
    return changed


def verify_existing(folder, manifest):
    for video, saved in manifest['videos'].items():
        shard, row = video.rsplit(':', 1)
        if read_part(folder / 'videos' / shard / f'{int(row):05d}.jsonl.gz') != saved:
            raise ValueError('Existing checkpoint metadata differs: ' + video)
    for shard, saved in manifest['shards'].items():
        if digest(folder / (shard + '.jsonl.gz')) != saved['sha256']:
            raise ValueError('Existing shard hash differs: ' + shard)


def worker(args):
    # CUDA_VISIBLE_DEVICES is set by the coordinator before importing PyTorch.
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from video_indexer.pipeline import Detector, video_bytes, video_records
    from video_indexer.decode import decoded_frames
    folder = args.checkpoints
    initial = json.loads((folder / 'manifest.json').read_text())
    if initial['signature'] != legacy_signature(args.revision):
        raise ValueError('Worker source/config differs from validated checkpoints')
    detector = Detector(ROOT / 'models/yolo11l.pt', '0', 'fp16')
    for index in map(int, args.shards.split(',')):
        shard = f'train-{index:05d}-of-01357.parquet'
        if shard in initial['shards'] or (folder / (shard + '.shard-metrics.json')).exists():
            continue
        if time.time() >= args.deadline:
            return
        source = Path(hf_hub_download('HuggingFaceFV/finevideo', repo_type='dataset',
                      filename='data/' + shard, revision=args.revision, local_dir=ROOT / 'finevideo'))
        parquet = pq.ParquetFile(source)
        iterator = iter(parquet.iter_batches(batch_size=1, columns=['mp4'], use_threads=False))
        read_failure = None
        members = []
        for row in range(parquet.metadata.num_rows):
            if time.time() >= args.deadline:
                return  # Interrupted shards reuse all fully saved videos next session.
            try:
                batch = next(iterator) if read_failure is None else None
            except Exception as exc:
                read_failure, batch = type(exc).__name__, None
            video = f'{shard}:{row}'

            def produce(metrics):
                if read_failure is not None:
                    yield {'video': video, 'error': 'unreadable Parquet data: ' + read_failure}
                    return
                try:
                    with tempfile.TemporaryDirectory(prefix='alethia-') as directory:
                        movie = Path(directory) / 'video.mp4'
                        movie.write_bytes(video_bytes(batch.column(0)[0].as_py()))
                        yield from video_records(video, decoded_frames(movie, 2), detector, 8, metrics)
                except Exception as exc:
                    yield {'video': video, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}

            part = folder / 'videos' / shard / f'{row:05d}.jsonl.gz'
            saved = save_video(part, produce)
            members.append(saved)
            print(json.dumps({'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'video': video, 'seconds': saved.get('seconds', 0),
                              'errors': saved.get('errors', 0)}), flush=True)
        output = folder / (shard + '.jsonl.gz')
        with output.with_suffix('.gz.tmp').open('wb') as stream:
            for row in range(parquet.metadata.num_rows):
                with (folder / 'videos' / shard / f'{row:05d}.jsonl.gz').open('rb') as member:
                    shutil.copyfileobj(member, stream)
        output.with_suffix('.gz.tmp').replace(output)
        totals = {key: sum(m.get(key, 0) for m in members)
                  for key in ('videos', 'seconds', 'errors', 'frames', 'wall_seconds')}
        atomic_json(folder / (shard + '.shard-metrics.json'),
                    {**totals, 'expected_videos': parquet.metadata.num_rows, 'sha256': digest(output)})
        source.unlink()


def coordinate(args):
    from scripts.capture_hardware import capture
    from scripts.audit_results import audit
    import torch
    folder = args.checkpoints
    folder.mkdir(parents=True, exist_ok=True)
    # OS lock is released on process death; stale lock files do not block recovery.
    with (folder / '.parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / 'manifest.json'
        signature = legacy_signature(args.revision)
        manifest = json.loads(path.read_text()) if path.exists() else {
            'signature': signature, 'started_utc': now(), 'dataset_revision': args.revision,
            'hardware': capture(), 'shards': {}, 'videos': {}, 'sessions': [],
            'inference_config': {'precision': 'fp16', 'batch_size': 8, 'decode_threads': 2}}
        if manifest['signature'] != signature:
            raise ValueError('Source/model/config differs from validated v2 checkpoints')
        devices = [int(x) for x in args.devices.split(',')]
        if len(set(devices)) != len(devices) or any(d < 0 or d >= torch.cuda.device_count() for d in devices):
            raise ValueError('Devices must be distinct available CUDA GPU indices')
        verify_existing(folder, manifest)
        collect(folder, manifest)
        session = {'started_utc': now(), 'hardware': capture(), 'devices': devices,
                   'runner': 'parallel_run.py', 'runner_sha256': digest(Path(__file__)),
                   'partition': 'shard_index modulo GPU count; one manifest writer'}
        manifest['sessions'].append(session)
        atomic_json(path, manifest)
        started = time.perf_counter()
        deadline = time.time() + args.max_hours * 3600
        processes, logs = [], []
        try:
            for slot, device in enumerate(devices):
                shards = ','.join(str(i) for i in range(slot, 50, len(devices)))
                log = (folder / f'gpu-{device}.log').open('a')
                logs.append(log)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device), 'MPLBACKEND': 'Agg'}
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                           '--checkpoints', str(folder.resolve()), '--revision', args.revision,
                           '--shards', shards, '--deadline', str(deadline)]
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
            while True:
                collect(folder, manifest)
                session['active_wall_seconds'] = time.perf_counter() - started
                atomic_json(path, manifest)
                exits = [p.poll() for p in processes]
                print(json.dumps({'utc': now(), 'videos': len(manifest['videos']),
                                  'shards': len(manifest['shards']),
                                  'errors': sum(v.get('errors', 0) for v in manifest['videos'].values()),
                                  'worker_exits': exits}), flush=True)
                if any(exit_code not in (None, 0) for exit_code in exits):
                    raise RuntimeError('A GPU worker failed; inspect gpu-*.log; no completion claimed')
                if all(exit_code is not None for exit_code in exits):
                    break
                time.sleep(30)
            collect(folder, manifest)
            session.update(finished_utc=now(), active_wall_seconds=time.perf_counter() - started)
            if len(manifest['shards']) != 50:
                session['stopped_at_session_budget'] = True
                atomic_json(path, manifest)
                print('PAUSED_WITH_CHECKPOINTS', flush=True)
                return
            final = folder / 'results.jsonl.gz'
            with final.with_suffix('.gz.tmp').open('wb') as stream:
                for index in range(50):
                    with (folder / f'train-{index:05d}-of-01357.parquet.jsonl.gz').open('rb') as member:
                        shutil.copyfileobj(member, stream)
            final.with_suffix('.gz.tmp').replace(final)
            complete = {**manifest, 'completed_utc': now(), 'completed_shards': 50,
                        'full_50_shards': True, 'results_sha256': digest(final)}
            complete['audit'] = audit(final, complete)
            atomic_json(path, complete)
            print('COMPLETE_AUDITED', json.dumps(complete['audit']), flush=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--max-hours', type=float, default=10)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--shards', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 10.5:
        parser.error('max-hours must be in (0, 10.5] to leave time to preserve session output')
    if args.worker:
        if not args.shards or args.deadline is None:
            parser.error('Worker requires shards and deadline')
        worker(args)
    else:
        coordinate(args)


if __name__ == '__main__':
    main()
