import gzip
import json
import pytest
from scripts.resumable_run import save_video


def test_interrupted_video_is_recomputed_but_finished_video_is_reused(tmp_path):
    path = tmp_path / "0.jsonl.gz"
    def interrupted(metrics):
        yield {"video": "shard:0", "second": 0}
        raise RuntimeError("runtime lost")
    with pytest.raises(RuntimeError):
        save_video(path, interrupted)
    assert not path.exists()
    records = [{"video": "shard:0", "second": n} for n in range(3)]
    saved = save_video(path, lambda metrics: iter(records))
    assert saved["seconds"] == 3
    def must_not_run(metrics):
        raise AssertionError("completed video must not be inferred twice")
    assert save_video(path, must_not_run) == saved
    with gzip.open(path, "rt") as f:
        assert [json.loads(line) for line in f] == records
    path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="modified"):
        save_video(path, must_not_run)
