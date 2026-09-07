import datetime
import uuid


def test_edition_is_atomic_idempotent_and_profile_scoped(db):
    from theseus_insight.data_access.newsletters import NewsletterRepository
    task_id = str(uuid.uuid4())
    artifact = {'papers': [{'key': 'fixture-paper'}]}
    ids = []
    try:
        first = NewsletterRepository.save_edition(task_id, 'Validated issue', datetime.date(2026, 1, 1), datetime.date(2026, 1, 2), [991], artifact)
        ids.append(first)
        second = NewsletterRepository.save_edition(task_id, 'Retry content', datetime.date(2026, 1, 1), datetime.date(2026, 1, 2), [991], artifact)
        assert first == second
        assert db.execute('SELECT content FROM newsletters WHERE id=%s', (first,)).fetchone()['content'] == 'Validated issue'
        assert 'fixture-paper' in NewsletterRepository.recent_paper_keys([991])
        assert 'fixture-paper' not in NewsletterRepository.recent_paper_keys([992])
        assert 'fixture-paper' not in NewsletterRepository.recent_paper_keys([])
    finally:
        for ident in ids:
            db.execute('DELETE FROM newsletters WHERE id=%s', (ident,))


async def test_content_checkpoint_recovery_still_persists_edition(db):
    from types import SimpleNamespace
    from theseus_insight.pipeline.stages import newsletter_content
    task_id = str(uuid.uuid4())
    async def load(name): return 'Recovered content' if name == 'newsletter_content' else None
    ti = SimpleNamespace(task_id=task_id, db_saving=True, start_date=datetime.date(2026, 1, 1), end_date=datetime.date(2026, 1, 2), _load_checkpoint_async=load)
    try:
        await newsletter_content.run(ti, {'papers': [], 'sections': []}, None, None)
        assert db.execute('SELECT newsletter_id FROM newsletter_editions WHERE task_id=%s', (task_id,)).fetchone()
    finally:
        db.execute('DELETE FROM newsletters WHERE id IN (SELECT newsletter_id FROM newsletter_editions WHERE task_id=%s)', (task_id,))
