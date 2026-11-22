"""Export this Codex chat's visible messages/tool records, redacting credentials."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PATTERNS = [re.compile(pattern) for pattern in (
    r"\bhf_[A-Za-z0-9]{20,}\b", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}\b", r"\bsk-[A-Za-z0-9_-]{20,}\b",
    # Kaggle embeds an authenticated editor session in observed iframe URLs.
    # Include truncated values, since accessibility output can shorten long URLs.
    r"(?<=session=)eyJ[A-Za-z0-9_.%-]+",
)]
VISIBLE_TOOLS = {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}


def redact(text):
    count = 0
    for pattern in PATTERNS:
        text, matches = pattern.subn("[REMOVED]", text)
        count += matches
    return text, count


def visible_item(payload):
    kind = payload.get("type")
    if kind == "message":
        if payload.get("role") not in {"user", "assistant"}:
            return False
        if payload.get("role") == "assistant" and payload.get("phase") not in {"commentary", "final_answer", "final"}:
            return False
        return True
    return kind in VISIBLE_TOOLS


def export(source, destination):
    counts = Counter()
    redactions = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    last_timestamp = None
    with source.open() as incoming, temporary.open("w") as outgoing:
        for line in incoming:
            record = json.loads(line)
            payload = record.get("payload", {})
            if record.get("type") != "response_item" or not visible_item(payload):
                continue
            # Runtime metadata and hidden/internal model records are not visible chat.
            item = {key: value for key, value in payload.items() if not key.startswith("internal_")}
            text, number = redact(json.dumps({"timestamp": record.get("timestamp"), "item": item}, ensure_ascii=False))
            outgoing.write(text + "\n")
            redactions += number
            counts[item.get("role", item["type"])] += 1
            last_timestamp = record.get("timestamp")
    temporary.replace(destination)
    return {"exported_utc": datetime.now(timezone.utc).isoformat(), "through_timestamp": last_timestamp,
            "records": sum(counts.values()), "counts": dict(counts), "credential_redactions": redactions,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "scope": "Visible user/assistant messages and tool calls/results from this local Codex session; no summaries or shortened tool content.",
            "refresh_required": "Rerun before submission to include subsequent chat activity; add other AI tool histories if any were used."}


def main():
    thread = os.environ.get("CODEX_THREAD_ID")
    if not thread:
        raise SystemExit("Run from the current Codex chat so CODEX_THREAD_ID identifies its own log.")
    sources = list((Path.home() / ".codex/sessions").rglob("*" + thread + "*.jsonl"))
    if len(sources) != 1:
        raise SystemExit("Could not uniquely locate this chat's session log")
    folder = Path(__file__).resolve().parents[1] / "ai_chats"
    manifest = export(sources[0], folder / "codex-visible.jsonl")
    manifest["thread_id"] = thread
    (folder / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
