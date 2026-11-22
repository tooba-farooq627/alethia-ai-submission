import gzip
import json

import pytest

from scripts.audit_results import audit


def record(video="example:0", second=0):
    return {"video": video, "second": second, "objects": ["person"], "people": 0}


def write(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


def test_error_can_follow_completed_seconds(tmp_path):
    path = write(tmp_path / "results.jsonl.gz", [record(), {"video": "example:0", "error": "truncated"}])
    assert audit(path) == {"valid": True, "videos": 1, "seconds": 1, "errors": 1,
                           "full_coverage_checked": False}


@pytest.mark.parametrize("rows", [
    [record(second=1)],
    [record(), record()],
    [record(), record(video="example:1"), record(second=1)],
    [{**record(), "people": True}],
    [{**record(), "objects": ["cat"]}],
    [{"video": "example:0", "error": "truncated"}, record()],
])
def test_rejects_broken_output(tmp_path, rows):
    with pytest.raises(ValueError):
        audit(write(tmp_path / "results.jsonl.gz", rows))


def test_complete_manifest_requires_every_video(tmp_path):
    names = [f"train-{i:05d}-of-01357.parquet" for i in range(50)]
    manifest = {"full_50_shards": True, "completed_shards": 50,
                "shards": {name: {"expected_videos": 1, "seconds": 1} for name in names}}
    path = write(tmp_path / "results.jsonl.gz", [record(video=name+":0") for name in names[:-1]])
    with pytest.raises(ValueError, match="1 missing"):
        audit(path, manifest)
    write(path, [record(video=name+":0") for name in names])
    assert audit(path, manifest)["full_coverage_checked"] is True
