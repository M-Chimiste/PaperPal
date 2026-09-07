import pytest
from fastapi.testclient import TestClient


def test_startup_readiness_and_shutdown(migrated_db, monkeypatch):
    from theseus_insight import main
    # Test lifespan and real dispatch against the isolated DB; never clean user media.
    monkeypatch.setattr(main, 'cleanup_old_media_files', lambda **kw: None)
    class Scheduler:
        is_running = False
        async def start(self): self.is_running = True
        async def stop(self): self.is_running = False
    monkeypatch.setattr(main, 'scheduler', Scheduler())
    with TestClient(main.app, base_url='http://localhost', client=('127.0.0.1', 50000)) as client:
        assert client.get('/health/live').json() == {'status': 'alive'}
        assert client.get('/health/ready').status_code == 200
    assert not main.app.state.ready
    assert main.task_manager.general_worker_task is None


def test_migration_failure_prevents_startup(migrated_db, monkeypatch):
    from theseus_insight import main
    from theseus_insight.db import migrations
    async def fail():
        raise RuntimeError('synthetic migration failure')
    monkeypatch.setattr(migrations, 'check_and_apply_migrations', fail)
    with pytest.raises(RuntimeError, match='startup failed'):
        with TestClient(main.app, base_url='http://localhost', client=('127.0.0.1', 50000)):
            pass
    assert not main.app.state.ready
