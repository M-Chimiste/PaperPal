"""Real newsletter handler, checkpoint adapter and delivery stage; fake inference/mail."""
import asyncio
import datetime
import uuid
from types import SimpleNamespace
import pytest


async def test_newsletter_recovery_does_not_resend(empty_db, tmp_path, monkeypatch):
    from theseus_insight.api.task_handlers import newsletter
    from theseus_insight.api.tasks import TaskManager
    from theseus_insight.pipeline.checkpoints import CheckpointAdapter
    from theseus_insight.pipeline.stages import email
    from theseus_insight.data_access import TaskRepository
    task_id = str(uuid.uuid4())
    sent = []
    generated = []
    attempts = []
    class FakePipeline:
        def __init__(self, **kwargs):
            self.task_id = kwargs['task_id']
            assert kwargs['checkpoint_dir'].endswith(task_id)
            self._checkpoints = CheckpointAdapter(str(tmp_path / task_id))
            self.generate_email, self.generate_podcast, self.verbose = True, False, False
            self.receiver_address = ['test@example.invalid']
            self.start_date = self.end_date = datetime.date(2026, 1, 1)
            self.communication = SimpleNamespace(compose_message=lambda *a: None, send_email=lambda: sent.append(task_id))
        async def _load_checkpoint_async(self, name):
            return self._checkpoints.load(name)
        def _log_error(self, *args):
            pass
        def run(self, progress_callback):
            async def execute():
                await self._checkpoints.init_db_job({'model': 'fake-v1', 'task_id': task_id})
                content = self._checkpoints.load('newsletter_content')
                if content is None:
                    generated.append(task_id)
                    self._checkpoints.save('newsletter_content', 'Mocked newsletter')
                    self._checkpoints.save('newsletter_sections', {'urls_and_titles': []})
                await email.run(self, None, None, None)
                attempts.append(task_id)
                if len(attempts) == 1:
                    raise RuntimeError('Simulated interruption after delivery')
                return {'newsletter': 'ready'}
            return asyncio.run(execute())
    monkeypatch.setattr(newsletter, 'TheseusInsight', FakePipeline)
    manager = TaskManager()
    await manager.create_task(task_id, 'newsletter', {})
    with pytest.raises(RuntimeError, match='Simulated interruption'):
        await newsletter.run(manager, task_id)
    TaskRepository.update_task_status(task_id, "pending")
    await newsletter.run(manager, task_id)
    assert sent == generated == [task_id]
    assert TaskRepository.get_task(task_id)['status'] == 'completed'


async def test_checkpoint_configuration_must_match(tmp_path):
    from theseus_insight.pipeline.checkpoints import CheckpointAdapter
    adapter = CheckpointAdapter(str(tmp_path))
    await adapter.init_db_job({'model': 'old'})
    adapter.save('papers_ranked', [1, 2])
    await CheckpointAdapter(str(tmp_path)).init_db_job({'model': 'old'})
    with pytest.raises(ValueError, match='configuration changed'):
        await CheckpointAdapter(str(tmp_path)).init_db_job({'model': 'new'})
