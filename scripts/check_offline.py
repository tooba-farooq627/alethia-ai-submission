"""Exercise the required CLI with Python outbound network operations denied."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_results import audit
from scripts.compare_results import compare

GUARD = '''import json, os, socket, sys
def offline_audit(event, arguments):
    if event == "socket.getaddrinfo" or (
        event == "socket.connect" and arguments[0].family in (socket.AF_INET, socket.AF_INET6)
    ):
        with open(os.environ["ALETHIA_NETWORK_ATTEMPTS"], "a") as stream:
            stream.write(json.dumps({"event": event}) + "\\n")
        raise OSError("Network disabled for offline verification")
sys.addaudithook(offline_audit)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = (args.output_dir / "results.jsonl").resolve()
    attempts = (args.output_dir / "network-attempts.jsonl").resolve()
    profile = (args.output_dir / "profile.json").resolve()
    with tempfile.TemporaryDirectory(prefix="alethia-offline-") as directory:
        guard = Path(directory)
        (guard / "sitecustomize.py").write_text(GUARD)
        environment = {**os.environ, "PYTHONPATH": str(guard) + os.pathsep + str(ROOT),
                       "ALETHIA_NETWORK_ATTEMPTS": str(attempts), "YOLO_OFFLINE": "true"}
        # Confirm the guard really prevents a connection before testing the CLI.
        proof = subprocess.run([sys.executable, "-c",
            "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)"],
            env=environment, capture_output=True, text=True)
        if proof.returncode == 0 or not attempts.exists():
            raise RuntimeError("network guard did not block its self-check")
        attempts.unlink()
        command = [sys.executable, str(ROOT / "run.py"), "--input", str(args.input.resolve()),
                   "--output", str(output), "--profile", str(profile)]
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        (args.output_dir / "stderr.log").write_text(result.stderr)
        (args.output_dir / "stdout.log").write_text(result.stdout)
        result.check_returncode()
    comparison = compare(args.reference, output)
    report = {"platform": sys.platform, "python": sys.version.split()[0],
              "guard_self_check_passed": True, "python_network_denied": True,
              "network_attempts": len(attempts.read_text().splitlines()) if attempts.exists() else 0,
              "audit": audit(output), "comparison": comparison,
              "profile": json.loads(profile.read_text()),
              "limitations": "Host check only; this is not a fresh Linux VM, OS-wide network isolation, or an RTX 4090 benchmark."}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not comparison["passes_diagnostic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
