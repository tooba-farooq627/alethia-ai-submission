"""GPU throughput probe. Sampling is for this benchmark only, never submission runs."""
import argparse
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import av
    import pyarrow.parquet as pq
    import torch
    from video_indexer.decode import decoded_frames
    from video_indexer.pipeline import Detector
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")
    if not torch.cuda.is_available():
        parser.error("an NVIDIA GPU is required")
    groups = {}
    for shard in sorted(args.input.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, columns=["mp4"]):
            raw = batch.column(0)[0].as_py()
            with av.open(io.BytesIO(raw)) as container:
                total = container.streams.video[0].frames
            stride = max(1, total // args.frames_per_video)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "video.mp4"
                path.write_bytes(raw)
                taken = 0
                for i, (_, image) in enumerate(decoded_frames(path)):
                    if i % stride == 0:
                        groups.setdefault(image.shape, []).append(image)
                        taken += 1
                        if taken >= args.frames_per_video:
                            break
    images = max(groups.values(), key=len)
    print(f"Benchmark frames: {len(images)}, shape: {images[0].shape}", flush=True)
    reports = []
    reference = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for precision, sizes in [("fp32", [1, 8, 32]), ("fp16", [8, 32, 64])]:
        detector = Detector(ROOT / "models/yolo11l.pt", "0", precision)
        for size in sizes:
            try:
                detector(images[:size])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                counts = []
                for i in range(0, len(images), size):
                    counts.extend(detector(images[i:i+size]))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if reference is None:
                    reference = counts
                report = {"precision": precision, "batch_size": size,
                          "frames": len(images), "wall_seconds": elapsed,
                          "fps": len(images) / elapsed,
                          "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                          "frame_count_agreement": sum(a == b for a, b in zip(reference, counts)) / len(images)}
            except torch.cuda.OutOfMemoryError:
                report = {"precision": precision, "batch_size": size, "error": "CUDA out of memory"}
                torch.cuda.empty_cache()
            reports.append(report)
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        del detector
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
