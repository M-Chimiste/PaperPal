"""Pin the B6 TaskManager/handler split: registry, delegates, _common helpers."""
import inspect


EXPECTED = {
    "research", "custom_newsletter", "profile_newsletter", "star-map", "profile_interest", "newsletter", "podcast", "visualizer", "database_export", "database_import",
    "mindmap_expand", "mindmap_pdf_parse", "profile_aware_ingest", "bulk_embed",
}


def test_registry_complete_and_async():
    from theseus_insight.api.task_handlers import HANDLERS

    assert set(HANDLERS) == EXPECTED
    for name, handler in HANDLERS.items():
        assert inspect.iscoroutinefunction(handler), f"{name} handler not async"
        params = list(inspect.signature(handler).parameters)
        assert params == ["task_manager", "task_id"], f"{name}: {params}"


def test_taskmanager_delegates_exist():
    """Routers pass these bound methods to enqueue_task — they must survive."""
    from theseus_insight.api.tasks import task_manager

    for method in [
        "run_newsletter_task", "run_podcast_task", "run_visualizer_task",
        "run_database_export_task", "run_database_import_task",
        "run_mindmap_expand_task", "run_mindmap_pdf_parse_task",
        "run_profile_aware_ingest_task", "run_bulk_embed_task",
    ]:
        assert inspect.iscoroutinefunction(getattr(task_manager, method)), method


def test_get_orchestration_config_db_and_file_fallback(empty_db):
    from theseus_insight.api.task_handlers._common import get_orchestration_config

    # DB value wins when present
    empty_db.execute(
        """INSERT INTO settings (key, value) VALUES ('orchestration', '{"judge_model": {"model_name": "x"}}')
           ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"""
    )
    assert get_orchestration_config()["judge_model"]["model_name"] == "x"

    # With no DB value, falls back to config/orchestration.json at the repo
    # root — exercises the parents[3] path from the new module location.
    empty_db.execute("DELETE FROM settings WHERE key = 'orchestration'")
    cfg = get_orchestration_config()
    assert "embedding_model" in cfg, "config-file fallback did not load"


def test_progress_callback_returns_callable():
    from theseus_insight.api.task_handlers._common import progress_callback
    from theseus_insight.api.tasks import task_manager

    cb = progress_callback(task_manager, "test-task-id")
    assert callable(cb)
