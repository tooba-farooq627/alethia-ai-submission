"""All-frame full run with atomic, verified checkpoints after every video."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.full_run import digest, now


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_video(destination, produce):
    """Never mark a video reusable until its entire gzip member is durable."""
    metadata = destination.with_suffix(".metrics.json")
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if not destination.exists() or digest(destination) != saved["sha256"]:
            raise ValueError(f"checkpoint missing or modified: {destination}")
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics = defaultdict(float)
    started = time.perf_counter()
    # Compress locally, then copy once to Drive; no per-second remote writes.
    with tempfile.TemporaryDirectory(prefix="alethia-checkpoint-") as directory:
        local = Path(directory) / "video.jsonl.gz"
        with gzip.open(local, "wt", encoding="utf-8") as output:
            for record in produce(metrics):
                output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
                metrics["errors" if "error" in record else "seconds"] += 1
        temporary = destination.with_suffix(".gz.tmp")
        shutil.copyfile(local, temporary)
        temporary.replace(destination)
    saved = {**metrics, "videos": 1, "wall_seconds": time.perf_counter() - started,
             "sha256": digest(destination), "completed_utc": now()}
    atomic_json(metadata, saved)
    return saved


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from video_indexer.pipeline import Detector, video_bytes, video_records
    from video_indexer.decode import decoded_frames
    from scripts.capture_hardware import capture
    from scripts.audit_results import audit
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        parser.error("count must be 1..50")
    folder = args.checkpoints
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "video_indexer").glob("*.py")) + [Path(__file__), ROOT / "requirements.txt", ROOT / "models/yolo11l.pt"]
    signature = hashlib.sha256(json.dumps({"sources": {str(p.relative_to(ROOT)): digest(p) for p in files},
                                          "revision": args.revision, "count": args.count,
                                          "batch_size": 8, "precision": "fp16", "decode_threads": 2}, sort_keys=True).encode()).hexdigest()
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "signature": signature, "started_utc": now(), "dataset_revision": args.revision,
        "hardware": capture(), "shards": {}, "videos": {}, "sessions": [],
        "inference_config": {"precision": "fp16", "batch_size": 8, "decode_threads": 2}}
    if manifest["signature"] != signature:
        raise ValueError("checkpoint source/config differs; use a new folder")
    session = {"started_utc": now(), "hardware": capture()}
    session_started = time.perf_counter()
    manifest["sessions"].append(session)
    atomic_json(path, manifest)
    detector = Detector(ROOT / "models/yolo11l.pt", "0", "fp16")
    for shard_index in range(args.count):
        filename = f"train-{shard_index:05d}-of-01357.parquet"
        shard_output = folder / (filename + ".jsonl.gz")
        if filename in manifest["shards"]:
            if not shard_output.exists() or digest(shard_output) != manifest["shards"][filename]["sha256"]:
                raise ValueError("shard checkpoint changed: " + filename)
            continue
        source = Path(hf_hub_download("HuggingFaceFV/finevideo", repo_type="dataset", filename="data/" + filename,
                                      revision=args.revision, local_dir=ROOT / "finevideo"))
        parquet = pq.ParquetFile(source)
        iterator = iter(parquet.iter_batches(batch_size=1, columns=["mp4"], use_threads=False))
        read_failure = None
        for row in range(parquet.metadata.num_rows):
            try:
                batch = next(iterator) if read_failure is None else None
            except Exception as exc:
                read_failure = type(exc).__name__
                batch = None
            video = f"{filename}:{row}"

            def produce(metrics):
                if read_failure is not None:
                    yield {"video": video, "error": "unreadable Parquet data: " + read_failure}
                    return
                try:
                    with tempfile.TemporaryDirectory(prefix="alethia-") as directory:
                        movie = Path(directory) / "video.mp4"
                        movie.write_bytes(video_bytes(batch.column(0)[0].as_py()))
                        yield from video_records(video, decoded_frames(movie, 2), detector, 8, metrics)
                except Exception as exc:
                    yield {"video": video, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

            part = folder / "videos" / filename / f"{row:05d}.jsonl.gz"
            saved = save_video(part, produce)
            manifest["videos"][video] = saved
            manifest["updated_utc"] = now()
            session["active_wall_seconds"] = time.perf_counter() - session_started
            atomic_json(path, manifest)
            print(json.dumps({"video": video, "completed_videos": len(manifest["videos"]),
                              "seconds": saved.get("seconds", 0), "errors": saved.get("errors", 0)}), flush=True)
        with shard_output.with_suffix(".gz.tmp").open("wb") as output:
            for row in range(parquet.metadata.num_rows):
                with (folder / "videos" / filename / f"{row:05d}.jsonl.gz").open("rb") as member:
                    shutil.copyfileobj(member, output)
        shard_output.with_suffix(".gz.tmp").replace(shard_output)
        totals = {key: sum(manifest["videos"][f"{filename}:{row}"].get(key, 0) for row in range(parquet.metadata.num_rows))
                  for key in ("videos", "seconds", "errors", "frames", "wall_seconds")}
        manifest["shards"][filename] = {**totals, "expected_videos": parquet.metadata.num_rows, "sha256": digest(shard_output)}
        atomic_json(path, manifest)
        source.unlink()
    final = folder / "results.jsonl.gz"
    with final.with_suffix(".gz.tmp").open("wb") as output:
        for index in range(args.count):
            with (folder / f"train-{index:05d}-of-01357.parquet.jsonl.gz").open("rb") as part:
                shutil.copyfileobj(part, output)
    final.with_suffix(".gz.tmp").replace(final)
    manifest.update(completed_utc=now(), completed_shards=args.count, full_50_shards=args.count == 50,
                    results_sha256=digest(final))
    manifest["audit"] = audit(final, manifest if args.count == 50 else None)
    session.update(finished_utc=now(), active_wall_seconds=time.perf_counter() - session_started)
    atomic_json(path, manifest)
    print("COMPLETE", json.dumps(manifest["audit"]), flush=True)


if __name__ == "__main__":
    main()
