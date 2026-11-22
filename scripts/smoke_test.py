"""Small real-YOLO integration check; downloads only an official example image."""
from pathlib import Path
import argparse
import io
import json
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/smoke")
    args = parser.parse_args()
    import av
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from compare_results import compare
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    image_file = folder / "bus.jpg"
    if not image_file.exists():
        with urllib.request.urlopen("https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg", timeout=30) as response:
            image_file.write_bytes(response.read())
    image = cv2.imread(str(image_file))
    assert image is not None
    memory = io.BytesIO()
    with av.open(memory, mode="w", format="mp4") as movie:
        stream = movie.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)
    inputs = folder / "input"
    inputs.mkdir(exist_ok=True)
    pq.write_table(pa.table({"mp4": [memory.getvalue(), b"damaged video"], "json": ["{}", "{}"]}),
                   inputs / "smoke.parquet")
    outputs = []
    for size in (1, 3):
        output = folder / f"batch-{size}.jsonl"
        if output.exists():
            raise FileExistsError(f"Preserve earlier evidence or select a fresh smoke folder: {output}")
        subprocess.run([sys.executable, str(ROOT / "run.py"), "--input", str(inputs),
                        "--output", str(output), "--device", args.device, "--batch-size", str(size),
                        "--profile", str(folder / f"batch-{size}-profile.json")], check=True)
        outputs.append(output)
    result = compare(*outputs)
    records = [json.loads(line) for line in outputs[0].read_text().splitlines()]
    assert len(records) == 3 and records[-1]["video"] == "smoke.parquet:1" and "error" in records[-1]
    assert [r["second"] for r in records[:2]] == [0, 1]
    assert all("person" in r["objects"] and r["people"] > 0 for r in records[:2])
    assert result["passes_diagnostic"]
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
