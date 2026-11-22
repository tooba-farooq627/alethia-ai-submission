"""Required offline entry point. Every decoded frame is evaluated."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from video_indexer.pipeline import DEFAULT_PRECISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path(__file__).parent / "models/yolo11l.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0")
    # Match the batch size that passed the broader all-frame GPU comparison.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default=DEFAULT_PRECISION)
    parser.add_argument("--decode-threads", type=int, default=2)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--profile", type=Path, help="Optional machine-readable run timings")
    args = parser.parse_args()
    if args.batch_size < 1 or args.decode_threads < 1:
        parser.error("batch size and decode threads must be positive")
    if not shutil.which("ffprobe"):
        parser.error("ffprobe is required; install ffmpeg during preparation")
    files = sorted(args.input.rglob("*.parquet")) if args.input.is_dir() else []
    if not files:
        parser.error("input folder contains no .parquet files")
    if len({p.name for p in files}) != len(files):
        parser.error("duplicate Parquet basenames would create ambiguous video identifiers")
    if args.output.exists():
        parser.error("output already exists; choose a fresh output path to preserve run evidence")
    from video_indexer.pipeline import Detector, process_parquet
    started = time.perf_counter()
    detector = Detector(args.weights, args.device, args.precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"batch_size": args.batch_size, "device": detector.device,
              "precision": detector.precision, "files": [], "all_frames": True}
    with args.output.open("x", encoding="utf-8") as output:
        for path in files:
            try:
                metrics = process_parquet(path, output, detector, args.batch_size,
                                          args.decode_threads, args.temp_dir)
            except Exception as exc:
                # Without a readable footer/schema the number/identity of rows is
                # unknowable. Report a file-level failure, not fabricated video rows.
                metrics = {"file_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            report["files"].append({"file": path.name, **metrics})
            print(json.dumps(report["files"][-1]), file=sys.stderr, flush=True)
    report["wall_seconds"] = time.perf_counter() - started
    report["inference_frames"] = detector.frames
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}), file=sys.stderr)
    return 2 if any("file_error" in f for f in report["files"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
