"""Benchmark, verify all-frame outputs, then run only a passing faster configuration."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import audit
from scripts.compare_results import compare


def passes(result):
    return (result["passes_diagnostic"] and result["object_micro_f1"] >= .995
            and result["people_exact"] >= .99 and result["people_within_one"] >= .999
            and result["reference_errors"] == result["candidate_errors"] == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if not args.durable.is_dir():
        parser.error("persistent storage must already be mounted and created")
    tuning = args.durable / "optimization_v1"
    tuning.mkdir(exist_ok=True)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "revision": args.revision,
               "status": "benchmarking", "checks": []}

    def persist():
        temporary = tuning / "status.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(tuning / "status.json")

    def run(arguments, label):
        print(label, flush=True)
        with (tuning / (label + ".log")).open("a") as log:
            subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    persist()
    try:
        benchmark = tuning / "inference-benchmark.json"
        if not benchmark.exists():
            run(["scripts/benchmark_inference.py", "--input", "sample", "--output", str(benchmark)], "benchmark")
        candidates = [r for r in json.loads(benchmark.read_text())
                      if r.get("precision") == "fp16" and "error" not in r and r["peak_gpu_mib"] < 9000]
        if not candidates:
            raise RuntimeError("No safe FP16 benchmark configuration; inspect benchmark.log")
        fastest = max(candidates, key=lambda r: r["fps"])
        # Prefer the smaller batch when its throughput is within 5% of the fastest.
        selected = min((r for r in candidates if r["fps"] >= .95 * fastest["fps"]), key=lambda r: r["batch_size"])
        size = selected["batch_size"]
        summary.update(status="sample_validation", selected=selected)
        persist()
        print("SELECTED", json.dumps(selected), flush=True)
        reference = ROOT / "artifacts/finevideo-validation-v1/reference.jsonl"
        candidate = tuning / "sample-candidate.jsonl"
        if not candidate.exists():
            run(["run.py", "--input", "sample", "--output", str(candidate), "--batch-size", str(size),
                 "--precision", "fp16", "--profile", str(tuning / "sample-profile.json")], "sample_candidate")
        audit(candidate)
        agreement = compare(reference, candidate)
        summary["checks"].append({"stage": "three_complete_videos", **agreement})
        persist()
        if not passes(agreement):
            raise RuntimeError(f"FP16 sample correctness gate failed: {agreement}")
        profile = json.loads((tuning / "sample-profile.json").read_text())
        summary["sample_wall_seconds"] = profile["wall_seconds"]
        summary["sample_fps"] = profile["inference_frames"] / profile["wall_seconds"]
        summary["estimated_full_hours"] = 125 * profile["wall_seconds"] / 1079.8415479145813
        if summary["estimated_full_hours"] > 45:
            raise RuntimeError("Measured projection exceeds the 45-hour run budget; inspect timings before launching")
        print("SAMPLE_PASSED", json.dumps(agreement), "FULL_HOURS_ESTIMATE", summary["estimated_full_hours"], flush=True)
        summary["status"] = "broader_validation"
        persist()
        broader = ROOT / "validation12"
        source = ROOT / "finevideo/data/train-00000-of-01357.parquet"
        if not broader.exists():
            run(["scripts/make_subset.py", "--source", str(source), "--output", str(broader), "--rows", "12"], "make_validation12")
        broad_ref, broad_candidate = tuning / "validation12-reference.jsonl", tuning / "validation12-candidate.jsonl"
        if not broad_ref.exists():
            run(["run.py", "--input", str(broader), "--output", str(broad_ref), "--batch-size", "8",
                 "--precision", "fp32", "--profile", str(tuning / "validation12-reference-profile.json")], "broader_reference")
        if not broad_candidate.exists():
            run(["run.py", "--input", str(broader), "--output", str(broad_candidate), "--batch-size", str(size),
                 "--precision", "fp16", "--profile", str(tuning / "validation12-candidate-profile.json")], "broader_candidate")
        audit(broad_ref)
        audit(broad_candidate)
        agreement = compare(broad_ref, broad_candidate)
        summary["checks"].append({"stage": "twelve_complete_videos", **agreement})
        if not passes(agreement):
            raise RuntimeError(f"Broader correctness gate failed: {agreement}")
        summary["status"] = "full_run"
        checkpoint = args.durable / "first_full_optimized_v1"
        summary["checkpoints"] = str(checkpoint)
        persist()
        print("BROADER_PASSED", json.dumps(agreement), flush=True)
        print("STARTING_FULL_50", checkpoint, flush=True)
        # Full processing evaluates every frame; benchmark sampling is never used here.
        subprocess.run([sys.executable, "-u", "scripts/full_run.py", "--checkpoints", str(checkpoint),
                        "--count", "50", "--batch-size", str(size), "--precision", "fp16",
                        "--revision", args.revision], cwd=ROOT, check=True)
        summary["audit"] = audit(checkpoint / "results.jsonl.gz", json.loads((checkpoint / "manifest.json").read_text()))
        summary["status"] = "complete"
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        print("FULL_50_COMPLETE", json.dumps(summary["audit"]), flush=True)
    except BaseException as exc:
        summary.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        persist()
        print("STOPPED", summary["error"], flush=True)
        raise


if __name__ == "__main__":
    main()
