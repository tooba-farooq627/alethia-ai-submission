import json
import sys
import types
import zipfile
from types import SimpleNamespace

from scripts import watch_run


def test_fresh_host_creates_snapshot_before_first_status(tmp_path, monkeypatch):
    """A newly booted host's monotonic clock may be below ten minutes."""
    project = tmp_path / 'source'
    project.mkdir()
    for name in ('run.py', 'prepare.py', 'requirements.txt', 'requirements-dev.txt', 'pytest.ini'):
        (project / name).write_text('')
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'manifest.json').write_text(json.dumps({'videos': {}, 'shards': {}}))
    display = types.ModuleType('IPython.display')
    display.FileLink = str
    display.clear_output = lambda **kwargs: None
    display.display = lambda *args: None
    monkeypatch.setitem(sys.modules, 'IPython', types.ModuleType('IPython'))
    monkeypatch.setitem(sys.modules, 'IPython.display', display)
    monkeypatch.setattr(watch_run.time, 'monotonic', lambda: 100.0)
    monkeypatch.setattr(watch_run.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))
    observed = []

    def advance(_seconds):
        status = json.loads((tmp_path / 'alethia-status.json').read_text())
        assert status['state'] == 'running'
        with zipfile.ZipFile(tmp_path / 'alethia-recovery.zip') as archive:
            assert archive.testzip() is None
            assert json.loads(archive.read('manifest.json'))['videos'] == {}
        assert status['snapshot_bytes'] > 0
        observed.append(status)

    monkeypatch.setattr(watch_run.time, 'sleep', advance)
    exits = iter([None, 0])
    worker = SimpleNamespace(pid=42, poll=lambda: next(exits))
    watch_run.monitor(worker, project, checkpoint, tmp_path / 'run.log')
    assert len(observed) == 1
    assert json.loads((tmp_path / 'alethia-status.json').read_text())['state'] == 'paused_with_checkpoints'
