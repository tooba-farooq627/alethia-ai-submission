"""Colab-friendly, shard-checkpointed run. Downloads one required shard at a time."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    from huggingface_hub import hf_hub_download
    from video_indexer.pipeline import Detector, process_parquet, DEFAULT_BATCH_SIZE, DEFAULT_PRECISION
    from capture_hardware import capture
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="Choose persistent storage for surviving Colab disconnects")
    parser.add_argument("--cache", type=Path, default=Path("finevideo"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default=DEFAULT_PRECISION)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--revision", default="main", help="Pin a dataset commit for reproducible reruns")
    args = parser.parse_args()
    if not 1 <= args.count <= 50 or args.batch_size < 1 or args.decode_threads < 1:
        parser.error("count must be 1..50, batch size/decode threads positive")
    weights = ROOT / "models/yolo11l.pt"
    if not weights.is_file():
        parser.error("run python prepare.py first")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    inputs = sorted((ROOT / "video_indexer").glob("*.py")) + [
        ROOT / "requirements.txt", ROOT / "run.py", Path(__file__), weights]
    signature = hashlib.sha256(json.dumps({
        "files": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "batch_size": args.batch_size, "decode_threads": args.decode_threads,
        "device": args.device, "precision": args.precision, "dataset_revision": args.revision,
        "count": args.count,
    }, sort_keys=True).encode()).hexdigest()
    manifest_path = args.checkpoints / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            parser.error("checkpoint config/source differs; use a new checkpoint folder")
    else:
        manifest = {"signature": signature, "started_utc": now(), "hardware": capture(),
                    "dataset_revision": args.revision, "shards": {}, "sessions": []}

    def persist():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    session = {"started_utc": now(), "active_wall_seconds": 0, "hardware": capture()}
    manifest["sessions"].append(session)
    session_started = time.perf_counter()
    persist()
    detector = Detector(weights, args.device, args.precision)
    manifest["inference_config"] = {"batch_size": args.batch_size,
                                     "precision": detector.precision, "device": detector.device}
    persist()
    for index in range(args.count):
        filename = f"train-{index:05d}-of-01357.parquet"
        destination = args.checkpoints / (filename + ".jsonl.gz")
        saved = manifest["shards"].get(filename)
        if saved:
            if not destination.exists() or digest(destination) != saved["sha256"]:
                raise ValueError(f"checkpoint missing or modified: {filename}")
            print(f"Checkpoint already verified: {filename}", flush=True)
            continue
        started = time.perf_counter()
        shard = Path(hf_hub_download("HuggingFaceFV/finevideo", repo_type="dataset",
                                    filename="data/" + filename, revision=args.revision,
                                    local_dir=args.cache))
        download_seconds = time.perf_counter() - started
        temporary = destination.with_suffix(".gz.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as output:
            metrics = process_parquet(shard, output, detector, args.batch_size, args.decode_threads)
        temporary.replace(destination)
        manifest["shards"][filename] = {**metrics, "download_seconds": download_seconds,
                                         "sha256": digest(destination)}
        session["active_wall_seconds"] = time.perf_counter() - session_started
        session["updated_utc"] = now()
        persist()
        print(json.dumps({"file": filename, **manifest["shards"][filename]}), flush=True)
        # Only the just-downloaded disposable cache file is removed. Model/results stay.
        shard.unlink()

    # Concatenated gzip members form a valid gzip file and need no recompression.
    final = args.checkpoints / "results.jsonl.gz"
    temporary = final.with_suffix(".gz.tmp")
    with temporary.open("wb") as output:
        for index in range(args.count):
            part = args.checkpoints / f"train-{index:05d}-of-01357.parquet.jsonl.gz"
            with part.open("rb") as source:
                import shutil
                shutil.copyfileobj(source, output)
    temporary.replace(final)
    session["active_wall_seconds"] = time.perf_counter() - session_started
    session["finished_utc"] = now()
    manifest["completed_utc"] = now()
    manifest["completed_shards"] = args.count
    manifest["full_50_shards"] = args.count == 50
    manifest["results_sha256"] = digest(final)
    persist()
    print(f"Complete: {args.count} shards -> {final}", flush=True)


if __name__ == "__main__":
    main()
