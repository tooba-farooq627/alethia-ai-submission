"""Download the official model before an offline timed run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

if __name__ == "__main__":
    from ultralytics import YOLO
    target = ROOT / "models/yolo11l.pt"
    target.parent.mkdir(exist_ok=True)
    model = YOLO(str(target), task="detect")
    assert model.names[0] == "person" and model.names[67] == "cell phone"
    print(f"Official YOLO11l model ready: {target}")
