import gzip
import json
import pytest
from scripts.finalize_results import finalize
from scripts.full_run import digest


def fixture(tmp_path, complete=True):
    folder = tmp_path / "run"
    folder.mkdir()
    names = [f"train-{i:05d}-of-01357.parquet" for i in range(50)]
    with gzip.open(folder / "results.jsonl.gz", "wt") as output:
        for name in names:
            output.write(json.dumps({"video": name+":0", "second": 0, "objects": [], "people": 0}) + "\n")
    hardware = {"gpu": "fixture GPU", "cpu": "fixture CPU", "logical_cpu_count": 2,
                "ram": "fixture RAM", "disks": "fixture disk", "os": "fixture OS", "python": "3.11"}
    manifest = {"full_50_shards": complete, "completed_shards": 50 if complete else 1,
        "started_utc": "2026-11-20T00:00:00+00:00", "completed_utc": "2026-11-20T01:00:00+00:00",
        "results_sha256": digest(folder / "results.jsonl.gz"),
        "sessions": [{"hardware": hardware, "active_wall_seconds": 3600}],
        "shards": {name: {"expected_videos": 1, "seconds": 1} for name in names}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_only_complete_audited_run_becomes_submission(tmp_path):
    source = fixture(tmp_path)
    destination = tmp_path / "submission"
    report = finalize(source, destination)
    assert report["full_coverage_checked"] and report["videos"] == 50
    assert digest(source / "results.jsonl.gz") == digest(destination / "results.jsonl.gz")
    assert "3600.000 seconds" in (destination / "HARDWARE.md").read_text()
    assert not (destination / "WRITEUP.md").exists()
    with gzip.open(source / "results.jsonl.gz", "at") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="hash"):
        finalize(source, destination)


def test_incomplete_run_never_creates_submission_results(tmp_path):
    source = fixture(tmp_path, complete=False)
    destination = tmp_path / "submission"
    with pytest.raises(ValueError, match="completed 50-shard"):
        finalize(source, destination)
    assert not destination.exists()
