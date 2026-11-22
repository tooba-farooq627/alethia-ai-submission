"""Streaming Parquet -> video -> inference -> second records."""
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .aggregate import CLASS_IDS, SecondAggregator

DEFAULT_BATCH_SIZE = 32
DEFAULT_PRECISION = "auto"


class Detector:
    def __init__(self, weights, device="auto", precision=DEFAULT_PRECISION):
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".runtime"))
        import torch
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Missing {weights}. Run python prepare.py before the offline run.")
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if precision not in {"auto", "fp32", "fp16"}:
            raise ValueError("precision must be auto, fp32, or fp16")
        cuda_device = self.device not in {"cpu", "mps"} and torch.cuda.is_available()
        self.precision = ("fp16" if cuda_device else "fp32") if precision == "auto" else precision
        if self.precision == "fp16" and not cuda_device:
            raise ValueError("FP16 inference requires a CUDA device")
        self.model = YOLO(str(weights), task="detect")
        self.frames = 0

    def __call__(self, images):
        counts = self._infer(images)
        self.frames += len(images)
        return counts

    def _infer(self, images):
        import torch
        try:
            return self._counts(images)
        except torch.cuda.OutOfMemoryError:
            if len(images) <= 1:
                raise
            torch.cuda.empty_cache()
        # Retry the same frames in order after leaving the exception scope.
        # This also handles larger square/portrait frames later in the dataset.
        middle = len(images) // 2
        return self._infer(images[:middle]) + self._infer(images[middle:])

    def _counts(self, images):
        import numpy as np
        # Retain all COCO classes through NMS/max_det to match official defaults.
        # Filter the five requested classes only after standard postprocessing.
        results = self.model.predict(images, imgsz=640, conf=0.4, iou=0.7,
                                     agnostic_nms=False, max_det=300, rect=True,
                                     half=self.precision == "fp16", augment=False, device=self.device,
                                     verbose=False, save=False, stream=False)
        counts = []
        for result in results:
            labels = result.boxes.cls.to("cpu").numpy().astype(np.int64)
            histogram = np.bincount(labels, minlength=80)
            counts.append(tuple(int(histogram[i]) for i in CLASS_IDS))
        if len(counts) != len(images):
            raise RuntimeError("model returned an unexpected number of frames")
        return counts


def video_records(video, frames, detector, batch_size=8, timings=None):
    """Emit completed seconds even if decoding later fails; never drop good batches."""
    timings = timings if timings is not None else defaultdict(float)
    aggregate = SecondAggregator(video)
    pending = []
    frame_count = 0

    def flush():
        nonlocal frame_count
        if not pending:
            return
        start = time.perf_counter()
        counts = detector([image for _, image in pending])
        timings["inference_seconds"] += time.perf_counter() - start
        if len(counts) != len(pending):
            raise RuntimeError("wrong number of detector results")
        start = time.perf_counter()
        records = []
        for (second, _), detected in zip(pending, counts):
            records.extend(aggregate.add(second, detected))
            frame_count += 1
        pending.clear()
        timings["aggregation_seconds"] += time.perf_counter() - start
        yield from records

    failure = None
    iterator = iter(frames)
    try:
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                timings["decode_seconds"] += time.perf_counter() - start
                break
            except Exception as exc:
                timings["decode_seconds"] += time.perf_counter() - start
                failure = exc
                break
            timings["decode_seconds"] += time.perf_counter() - start
            if pending and getattr(pending[0][1], "shape", None) != getattr(item[1], "shape", None):
                yield from flush()
            pending.append(item)
            if len(pending) >= batch_size:
                yield from flush()
        yield from flush()
        if failure is not None:
            raise failure
        if not frame_count:
            raise ValueError("could not decode any video frames")
        yield from aggregate.finish()
    except Exception as exc:
        # A current second may be incomplete; only previously completed seconds survive.
        yield {"video": video, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
        timings["frames"] += frame_count


def video_bytes(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    # Also accept Hugging Face's embedded {bytes, path} representation.
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray, memoryview)):
        return bytes(value["bytes"])
    raise ValueError("mp4 column does not contain embedded video bytes")


def process_parquet(path, output, detector, batch_size=8, decode_threads=2, temp_dir=None):
    import pyarrow.parquet as pq
    from .decode import decoded_frames
    timings = defaultdict(float)
    started = time.perf_counter()
    parquet = pq.ParquetFile(path)
    row = 0
    timings["expected_videos"] = parquet.metadata.num_rows

    def write(record):
        start = time.perf_counter()
        output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
        timings["write_seconds"] += time.perf_counter() - start
        timings["errors" if "error" in record else "seconds"] += 1

    iterator = iter(parquet.iter_batches(batch_size=1, columns=["mp4"], use_threads=False))
    while True:
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            timings["parquet_read_seconds"] += time.perf_counter() - start
            break
        except Exception as exc:
            # A damaged row group can make the remaining rows unreadable. Each
            # remaining video still gets its own valid, zero-based error identifier.
            for failed in range(row, parquet.metadata.num_rows):
                write({"video": f"{path.name}:{failed}", "error": f"unreadable Parquet data: {type(exc).__name__}"})
            timings["videos"] += parquet.metadata.num_rows - row
            break
        timings["parquet_read_seconds"] += time.perf_counter() - start
        video = f"{path.name}:{row}"
        try:
            start = time.perf_counter()
            raw = video_bytes(batch.column(0)[0].as_py())
            timings["parquet_read_seconds"] += time.perf_counter() - start
            with tempfile.TemporaryDirectory(prefix="alethia-", dir=temp_dir) as directory:
                movie = Path(directory) / "video.mp4"
                start = time.perf_counter()
                movie.write_bytes(raw)
                del raw
                timings["temporary_io_seconds"] += time.perf_counter() - start
                for record in video_records(video, decoded_frames(movie, decode_threads), detector,
                                            batch_size, timings):
                    write(record)
        except Exception as exc:
            write({"video": video, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        row += 1
        timings["videos"] += 1
        output.flush()
    timings["wall_seconds"] = time.perf_counter() - started
    return dict(timings)
