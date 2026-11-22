"""Read-only readiness check; never reads or evaluates the independent WRITEUP."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import audit


def check(root, manifest_path=None):
    findings = []
    def add(name, passed, detail):
        findings.append({"check": name, "passed": bool(passed), "detail": detail})
    for name in ("run.py", "requirements.txt", "README.md", "HARDWARE.md"):
        add(name, (root / name).is_file(), "Required file exists")
    # Presence only: the brief forbids AI writing, rewriting, or checking answers.
    writeup = root / "WRITEUP.md"
    add("independent_writeup_present", writeup.is_file() and writeup.stat().st_size > 0,
        "Existence/size only; contents are never opened or assessed")
    requirements = root / "requirements.txt"
    unpinned = []
    if not requirements.exists():
        unpinned.append("missing")
    else:
        for line in requirements.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                specifiers = list(Requirement(line.strip()).specifier)
                if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
                    unpinned.append(line)
            except Exception:
                unpinned.append(line)
    add("exact_dependency_pins", not unpinned, "Every dependency line uses ==")
    result = root / "results.jsonl.gz"
    manifest = None
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be a JSON object")
        except Exception as exc:
            manifest = None
            add("readable_manifest", False, str(exc))
    if result.is_file() and manifest_path is not None:
        try:
            if manifest is None:
                raise ValueError("full-run manifest unavailable")
            from scripts.full_run import digest
            if digest(result) != manifest.get("results_sha256"):
                raise ValueError("results hash differs from full-run manifest")
            detail = audit(result, manifest)
            add("full_50_results", True, detail)
        except Exception as exc:
            add("full_50_results", False, str(exc))
    else:
        add("full_50_results", False, "Require results.jsonl.gz and --manifest from the completed full run")
    folder = root / "ai_chats"
    exports = [p for p in folder.glob("*") if p.is_file() and p.suffix in {".jsonl", ".txt", ".html"} and p.stat().st_size]
    add("ai_chat_export_present", bool(exports), [p.name for p in exports])
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=root, capture_output=True, text=True)
    add("repository_remote_configured", remote.returncode == 0, remote.stdout.strip() or "No origin remote; public visibility still requires verification")
    hardware = root / "HARDWARE.md"
    hardware_matches = bool(manifest and manifest.get("full_50_shards") and hardware.is_file()
        and manifest.get("results_sha256") and manifest["results_sha256"] in hardware.read_text())
    add("actual_full_run_hardware_timing", hardware_matches,
        "Finalize the complete manifest into HARDWARE.md; its result hash must match")
    return {"ready": all(item["passed"] for item in findings), "checks": findings,
            "scope": "Local file checks only. Public repository visibility, committed contents, chat privacy/completeness, and Linux/RTX4090 performance need separate verification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(ROOT, args.manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)
