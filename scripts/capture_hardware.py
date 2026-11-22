"""Write factual machine metadata; no claimed full-run timing is invented."""
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def capture():
    import torch
    return {
        "os": platform.platform(), "python": platform.python_version(),
        "cpu": command(["lscpu"]) or command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpu_count": os.cpu_count(),
        "ram": command(["free", "-b"]) or command(["sysctl", "-n", "hw.memsize"]),
        "disks": command(["lsblk", "-d", "-o", "NAME,MODEL,ROTA,SIZE"]),
        "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture(), indent=2) + "\n")
    print(args.output)
