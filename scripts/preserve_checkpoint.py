"""Verify a downloaded recovery ZIP before preparing the next-session notebook."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import audit
from scripts.parallel_run import legacy_signature


def preserve(source, output_base):
    folder = output_base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True)
    destination = folder / 'alethia-recovery.zip'
    shutil.copy2(source, destination)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    with zipfile.ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in bundle.namelist()):
            raise ValueError('Unsafe archive path')
        manifest = json.loads(bundle.read('manifest.json'))
        if manifest['signature'] != legacy_signature(manifest['dataset_revision']):
            raise ValueError('Downloaded checkpoint source/model/config differs')
        for video, metrics in manifest['videos'].items():
            shard, row = video.rsplit(':', 1)
            part = f'videos/{shard}/{int(row):05d}.jsonl.gz'
            if sha(bundle.read(part)) != metrics['sha256']:
                raise ValueError('Video hash differs: ' + video)
            if json.loads(bundle.read(part.removesuffix('.gz') + '.metrics.json')) != metrics:
                raise ValueError('Video metadata differs: ' + video)
        for shard, metrics in manifest['shards'].items():
            if sha(bundle.read(shard + '.jsonl.gz')) != metrics['sha256']:
                raise ValueError('Shard hash differs: ' + shard)
        if manifest.get('full_50_shards') and sha(bundle.read('results.jsonl.gz')) != manifest['results_sha256']:
            raise ValueError('Final result hash differs')
        contents = folder / 'contents'
        bundle.extractall(contents)
    complete = bool(manifest.get('full_50_shards'))
    final_audit = audit(contents / 'results.jsonl.gz', manifest) if complete else None
    summary = {'backup_utc': datetime.now(timezone.utc).isoformat(), 'archive': str(destination.resolve()),
               'contents': str(contents.resolve()), 'sha256': sha(destination.read_bytes()),
               'videos': len(manifest['videos']), 'shards': len(manifest['shards']),
               'seconds': sum(v.get('seconds', 0) for v in manifest['videos'].values()),
               'errors': sum(v.get('errors', 0) for v in manifest['videos'].values()),
               'updated_utc': manifest.get('updated_utc'), 'all_video_and_shard_hashes_verified': True,
               'full_50_complete': complete, 'final_audit': final_audit}
    (folder / 'verified-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    latest = ROOT / 'artifacts/kaggle-migration/latest-verified-backup.json'
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-base', type=Path, default=ROOT / 'checkpoints/kaggle')
    parser.add_argument('--prepare-notebook', action='store_true')
    args = parser.parse_args()
    summary = preserve(args.archive, args.output_base)
    if args.prepare_notebook:
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_kaggle.py'), '--checkpoint',
                        summary['contents'], '--output', str(ROOT / 'artifacts/kaggle-migration/Kaggle_Resume_Dual_GPU.ipynb'),
                        '--parallel'], check=True)
    print(json.dumps(summary, indent=2))
