import asyncio
import threading
import httpx
from fastapi import FastAPI
from theseus_insight.services import search_service


async def test_blocking_search_keeps_api_responsive_and_rejects_overload(monkeypatch):
    monkeypatch.setattr(search_service, '_slots', threading.BoundedSemaphore(1))
    entered, release = threading.Event(), threading.Event()
    app = FastAPI()
    @app.get('/search')
    @search_service.bounded_search
    def search():
        entered.set()
        release.wait(timeout=5)
        return {'ok': True}
    @app.get('/ping')
    async def ping():
        return {'ok': True}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        first = asyncio.create_task(client.get('/search'))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            response = await asyncio.wait_for(client.get('/ping'), .5)
            assert response.status_code == 200
            overloaded = await client.get('/search')
            assert overloaded.status_code == 429
            assert overloaded.headers['retry-after'] == '2'
        finally:
            release.set()
            await first
