"""Copy only an audited complete run into submission files, preserving old evidence."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import audit
from scripts.full_run import digest


def finalize(run_dir, output_dir):
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = run_dir / "results.jsonl.gz"
    result_hash = digest(source)
    if result_hash != manifest.get("results_sha256"):
        raise ValueError("results hash differs from full-run manifest")
    report = audit(source, manifest)  # Includes exact first-50 video coverage.
    sessions = manifest.get("sessions", [])
    inventory = [s["hardware"] for s in sessions if s.get("hardware")] or [manifest.get("hardware", {})]
    needed = ("gpu", "cpu", "logical_cpu_count", "ram", "disks", "os", "python")
    if any(not hardware.get(key) for hardware in inventory for key in needed):
        raise ValueError("full-run hardware inventory is incomplete")
    start = datetime.fromisoformat(manifest["started_utc"])
    end = datetime.fromisoformat(manifest["completed_utc"])
    if end < start:
        raise ValueError("full-run end precedes its start")
    if any("active_wall_seconds" not in session for session in sessions) or not sessions:
        raise ValueError("full-run session timings are missing")
    active = sum(s["active_wall_seconds"] for s in sessions)
    hardware_text = ["# Full-run hardware and measured timings\n",
        f"Dataset revision: `{manifest.get('dataset_revision', 'unrecorded')}`\n",
        f"Started (UTC): {manifest['started_utc']}\nCompleted (UTC): {manifest['completed_utc']}\n",
        f"Elapsed start-to-finish, including interruptions: {(end-start).total_seconds():.3f} seconds.\n",
        f"Recorded active session time, including downloads/checkpoint work: {active:.3f} seconds.\n",
        f"Completed shards: 50. Videos: {report['videos']}. Output seconds: {report['seconds']}. Video errors: {report['errors']}.\n",
        f"Results SHA-256: `{result_hash}`\n",
        "The first FP32 full run did not complete; no completed baseline full-run time is available. Subset projections are not full-run measurements.\n",
        "The following inventories are the actual recorded full-run sessions. Disk inventory is reported as observed; cloud backing storage is not assumed.\n"]
    for number, hardware in enumerate(inventory, 1):
        hardware_text.append(f"## Recorded session {number}\n")
        for key in needed:
            hardware_text.append(f"{key}:\n```text\n{hardware[key]}\n```\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "results.jsonl.gz"
    if destination.exists() and digest(destination) != result_hash:
        raise FileExistsError("different submitted results already exist; preserve them or choose a new output directory")
    if source.resolve() != destination.resolve() and not destination.exists():
        temporary = destination.with_suffix(".gz.tmp")
        shutil.copyfile(source, temporary)
        if digest(temporary) != result_hash:
            raise ValueError("copied result hash mismatch")
        temporary.replace(destination)
    hardware_file = output_dir / "HARDWARE.md"
    text = "\n".join(hardware_text)
    if hardware_file.exists() and hardware_file.read_text() != text:
        backup = output_dir / "artifacts/finalization-backups"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hardware_file, backup / ("HARDWARE-" + digest(hardware_file)[:12] + ".md"))
    hardware_file.write_text(text)
    (output_dir / "full-run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "final-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir, args.output_dir), indent=2))
