import json
from scripts.export_ai_chat import export, redact


def test_visible_chat_is_complete_and_credentials_are_redacted(tmp_path):
    source, output = tmp_path / "session.jsonl", tmp_path / "visible.jsonl"
    secret = "hf_" + "a" * 32
    payloads = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "reasoning", "encrypted_content": "private model data"},
        {"type": "message", "role": "developer", "content": []},
        {"type": "function_call_output", "output": "credential " + secret},
        {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "done"}]},
    ]
    source.write_text("".join(json.dumps({"type": "response_item", "payload": p, "timestamp": str(i)}) + "\n" for i, p in enumerate(payloads)))
    report = export(source, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["item"]["type"] for row in rows] == ["message", "function_call_output", "message"]
    assert rows[0]["item"] == payloads[0] and rows[-1]["item"] == payloads[-1]
    assert report["credential_redactions"] == 1
    assert secret not in output.read_text() and "[REMOVED]" in output.read_text()


def test_editor_session_credentials_are_removed_without_breaking_url_or_json():
    secret = 'eyJheader.eyJpayload.'
    message = {'url': 'https://example.test/editor?session=' + secret + '&lsp=true'}
    sanitized, count = redact(json.dumps(message))
    assert count == 1
    assert json.loads(sanitized)['url'] == 'https://example.test/editor?session=[REMOVED]&lsp=true'
    shortened, count = redact('session=eyJpartiallyTruncatedCredential')
    assert shortened == 'session=[REMOVED]' and count == 1
    assert redact('session=ordinary-example') == ('session=ordinary-example', 0)
