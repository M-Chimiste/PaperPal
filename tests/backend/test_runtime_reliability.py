import asyncio
import uuid
import pytest
from theseus_insight.data_access import TaskRepository
from theseus_insight.data_access.runtime import DispatchRepository, DeliveryRepository


def create(handler='newsletter'):
    task_id = str(uuid.uuid4())
    TaskRepository.insert_task(task_id, handler, 'pending', {})
    DispatchRepository.enqueue(task_id, handler, 'general')
    return task_id


def test_expired_lease_cannot_steal_live_lock(empty_db):
    task_id = create()
    with DispatchRepository.claim(task_id, 'first') as first:
        assert first['attempts'] == 1
        empty_db.execute("UPDATE task_dispatch SET lease_until=now()-interval '1 minute' WHERE task_id=%s", (task_id,))
        with DispatchRepository.claim(task_id, 'second') as other:
            assert other is None
    with DispatchRepository.claim(task_id, 'second') as recovered:
        assert recovered['attempts'] == 2
        DispatchRepository.finish(task_id, 'second')
    assert task_id not in DispatchRepository.candidates('general')


def test_delivery_requires_explicit_resolution_after_interruption(empty_db):
    task_id = create()
    assert DeliveryRepository.begin(task_id, task_id)
    with pytest.raises(RuntimeError, match='uncertain'):
        DeliveryRepository.begin(task_id, task_id)
    DeliveryRepository.finish(task_id, 'sent')
    assert not DeliveryRepository.begin(task_id, task_id)


async def test_dispatch_survives_manager_recreation(empty_db, monkeypatch):
    from theseus_insight.api.tasks import TaskManager, TaskStatus
    from theseus_insight.api.task_handlers import HANDLERS
    task_id = create()
    calls = []
    async def fake(manager, tid):
        calls.append(tid)
        await manager.update_task_status(tid, TaskStatus.COMPLETED)
    monkeypatch.setitem(HANDLERS, 'newsletter', fake)
    manager = TaskManager()
    assert calls == []
    await manager.start_worker()
    try:
        for _ in range(50):
            if calls:
                break
            await asyncio.sleep(.1)
    finally:
        await manager.stop_worker()
    assert calls == [task_id]
    assert TaskRepository.get_task(task_id)['status'] == 'completed'


def test_hybrid_retains_keyword_only_matches(seeded_data, db):
    from theseus_insight.data_access import PaperRepository
    class Model:
        def invoke(self, query):
            return [1.] + [0.] * 767
    result = PaperRepository.hybrid_search('obsolete', Model(), embedding_model_name='fake-model')
    assert 3 in [paper['id'] for paper in result['items']]
    assert result['count_scope'] == 'retrieved_candidates'
    other_model = PaperRepository.hybrid_search('nonexistent', Model(), embedding_model_name='different-model')
    assert other_model['items'] == []


def test_hybrid_profile_filters_apply_to_both_branches(seeded_data):
    from theseus_insight.data_access import PaperRepository
    result = PaperRepository.hybrid_search('graph', None, semantic_weight=0, keyword_weight=1,
        profile_ids=[seeded_data['test_profile_id']], min_profile_score=7)
    assert result['items'] == []
    result = PaperRepository.hybrid_search('transformers', None, semantic_weight=0, keyword_weight=1,
        profile_ids=[seeded_data['test_profile_id']], min_profile_score=7)
    assert [p['id'] for p in result['items']] == [1]
    assert result['items'][0]['profile_score'] == 8


def test_keyword_only_api_accepts_missing_embeddings(client, seeded_data, db, monkeypatch):
    from theseus_insight.api.routers import papers
    from theseus_insight.data_access import SettingsRepository
    import json
    db.execute("UPDATE papers SET embedding_model=NULL WHERE id=3")
    SettingsRepository.set('orchestration', json.dumps({'embedding_model': {'model_name': 'fake-model'}}))
    def forbidden(*args):
        raise AssertionError('Keyword-only search must not initialize an embedding model')
    monkeypatch.setattr(papers, 'get_search_model', forbidden)
    response = client.post('/api/papers/hybrid-search', json={'query_text': 'obsolete', 'semantic_weight': 0, 'keyword_weight': 1})
    assert response.status_code == 200
    assert response.json()['results'][0]['id'] == 3
