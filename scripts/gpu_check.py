"""Check real YOLO inference on a CUDA GPU using an official example image."""
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    import torch
    from video_indexer.pipeline import Detector
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this readiness check")
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    sample = folder / "gpu-check-bus.jpg"
    if not sample.exists():
        with urllib.request.urlopen("https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg", timeout=30) as response:
            sample.write_bytes(response.read())
    image = cv2.imdecode(np.frombuffer(sample.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the GPU check image")
    detector = Detector(ROOT / "models/yolo11l.pt", "0")
    one = detector([image])[0]
    batched = detector([image, image, image])
    assert one[0] > 0, "example image should contain detected people"
    assert batched == [one] * 3, "single-frame and batched detections differ"
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "frames": detector.frames, "counts_person_car_dog_laptop_phone": one,
              "batch_counts_match": True, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
    (folder / "gpu-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
