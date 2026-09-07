async def test_only_one_scheduler_leads(migrated_db, monkeypatch):
    from theseus_insight.scheduler import TheseusScheduler
    async def no_scheduled_tasks(self):
        pass
    monkeypatch.setattr(TheseusScheduler, '_sync_scheduled_tasks', no_scheduled_tasks)
    first, second = TheseusScheduler(), TheseusScheduler()
    try:
        await first.start()
        await second.start()
        assert first.is_running
        assert second.standby and not second.is_running
        await first.stop()
        await second.start()
        assert second.is_running and not second.standby
    finally:
        await first.stop()
        await second.stop()
