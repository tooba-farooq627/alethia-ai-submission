import gzip
import json

import pytest

from scripts.full_run import digest
from scripts.parallel_run import collect, verify_existing
from scripts.resumable_run import atomic_json


def checkpoint(folder, shard=1, row=0):
    name = f'train-{shard:05d}-of-01357.parquet'
    part = folder / 'videos' / name / f'{row:05d}.jsonl.gz'
    part.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(part, 'wt') as output:
        output.write(json.dumps({'video': f'{name}:{row}', 'second': 0, 'objects': [], 'people': 0}) + '\n')
    saved = {'videos': 1, 'seconds': 1, 'errors': 0, 'sha256': digest(part)}
    atomic_json(part.with_suffix('.metrics.json'), saved)
    return name, part, saved


def test_coordinator_recovers_completed_video_without_manifest_entry(tmp_path):
    shard, part, saved = checkpoint(tmp_path)
    state = {'videos': {}, 'shards': {}}
    assert collect(tmp_path, state)
    assert state['videos'] == {f'{shard}:0': saved}
    assert not collect(tmp_path, state)
    verify_existing(tmp_path, state)
    part.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Invalid video checkpoint'):
        verify_existing(tmp_path, state)


def test_coordinator_refuses_uncommitted_corruption(tmp_path):
    _, part, _ = checkpoint(tmp_path)
    part.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Invalid video checkpoint'):
        collect(tmp_path, {'videos': {}, 'shards': {}})


def test_coordinator_refuses_changed_committed_metadata(tmp_path):
    shard, part, saved = checkpoint(tmp_path)
    state = {'videos': {f'{shard}:0': saved}, 'shards': {}}
    atomic_json(part.with_suffix('.metrics.json'), {**saved, 'seconds': 2})
    with pytest.raises(ValueError, match='committed video metadata changed'):
        collect(tmp_path, state)


def test_shard_is_committed_only_with_consistent_members_and_totals(tmp_path):
    shard, part, saved = checkpoint(tmp_path)
    output = tmp_path / (shard + '.jsonl.gz')
    output.write_bytes(part.read_bytes())
    totals = {key: saved.get(key, 0) for key in ('videos', 'seconds', 'errors', 'frames', 'wall_seconds')}
    metadata = tmp_path / (shard + '.shard-metrics.json')
    atomic_json(metadata, {**totals, 'expected_videos': 1, 'sha256': digest(output)})
    state = {'videos': {}, 'shards': {}}
    assert collect(tmp_path, state)
    assert len(state['shards']) == 1
    state['shards'].clear()
    atomic_json(metadata, {**totals, 'seconds': 2, 'expected_videos': 1, 'sha256': digest(output)})
    with pytest.raises(ValueError, match='totals disagree'):
        collect(tmp_path, state)
