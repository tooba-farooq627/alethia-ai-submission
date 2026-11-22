"""Validate result structure and full-run video coverage without loading videos."""
import argparse
import gzip
import json
from pathlib import Path


OBJECTS = {"person", "car", "dog", "laptop", "cell phone"}


def audit(path, manifest=None):
    seen = set()
    current = None
    next_second = seconds = errors = 0
    failed = False
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            video = row.get("video")
            if not isinstance(video, str) or not video:
                raise ValueError(f"line {line_number}: invalid video identifier")
            if video != current:
                if video in seen:
                    raise ValueError(f"line {line_number}: repeated video group {video}")
                seen.add(video)
                current, next_second, failed = video, 0, False
            if failed:
                raise ValueError(f"line {line_number}: record after video error")
            if "error" in row:
                if set(row) != {"video", "error"} or not isinstance(row["error"], str) or not row["error"]:
                    raise ValueError(f"line {line_number}: invalid error record")
                errors += 1
                failed = True
                continue
            if set(row) != {"video", "second", "objects", "people"}:
                raise ValueError(f"line {line_number}: incorrect record fields")
            if type(row["second"]) is not int or row["second"] != next_second:
                raise ValueError(f"line {line_number}: expected second {next_second} for {video}")
            objects = row["objects"]
            if (not isinstance(objects, list) or any(not isinstance(x, str) for x in objects)
                    or len(set(objects)) != len(objects) or not set(objects) <= OBJECTS):
                raise ValueError(f"line {line_number}: invalid objects")
            if type(row["people"]) is not int or row["people"] < 0:
                raise ValueError(f"line {line_number}: invalid people count")
            if row["people"] and "person" not in objects:
                raise ValueError(f"line {line_number}: positive people count without person presence")
            next_second += 1
            seconds += 1
    if not seen:
        raise ValueError("empty results")
    if manifest is not None:
        if not manifest.get("full_50_shards") or manifest.get("completed_shards") != 50:
            raise ValueError("manifest is not a completed 50-shard run")
        expected_shards = {f"train-{i:05d}-of-01357.parquet" for i in range(50)}
        if set(manifest["shards"]) != expected_shards:
            raise ValueError("manifest does not contain exactly the first 50 shards")
        expected = {f"{name}:{i}" for name, metrics in manifest["shards"].items()
                    for i in range(int(metrics["expected_videos"]))}
        missing, extra = expected - seen, seen - expected
        if missing or extra:
            raise ValueError(f"video coverage mismatch: {len(missing)} missing, {len(extra)} unexpected")
        if errors != sum(int(m.get("errors", 0)) for m in manifest["shards"].values()):
            raise ValueError("error count differs from the run manifest")
        if seconds != sum(int(m.get("seconds", 0)) for m in manifest["shards"].values()):
            raise ValueError("second count differs from the run manifest")
    return {"valid": True, "videos": len(seen), "seconds": seconds, "errors": errors,
            "full_coverage_checked": manifest is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    print(json.dumps(audit(args.results, manifest), indent=2))
