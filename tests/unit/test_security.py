import base64
import hashlib
import pytest
from cryptography.fernet import InvalidToken
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from theseus_insight.security import encrypt, decrypt, migrate_legacy, AccessMiddleware


def test_authenticated_encryption_and_explicit_legacy_migration(monkeypatch):
    token = encrypt('secret-value')
    assert token != encrypt('secret-value')
    assert decrypt(token) == 'secret-value'
    with pytest.raises(ValueError, match='explicit migration'):
        decrypt('plaintext')
    key = hashlib.sha256(b'test_secret').digest()
    legacy = base64.b64encode(bytes(b ^ key[i % len(key)] for i, b in enumerate(b'legacy'))).decode()
    assert decrypt(migrate_legacy(legacy, 'xor')) == 'legacy'
    assert decrypt(migrate_legacy('plaintext', 'plaintext')) == 'plaintext'
    monkeypatch.setenv('APP_SECRET_KEY', 'wrong-key')
    with pytest.raises(InvalidToken):
        decrypt(token)


def test_no_default_encryption_key(monkeypatch):
    monkeypatch.delenv('APP_SECRET_KEY')
    with pytest.raises(ValueError):
        encrypt('secret')


def make_client(base_url='http://localhost'):
    async def index(request):
        return JSONResponse({'ok': True})
    async def ws(socket):
        await socket.accept()
        await socket.send_text('ok')
        await socket.close()
    app = Starlette(routes=[Route('/', index), WebSocketRoute('/ws', ws)])
    app.add_middleware(AccessMiddleware)
    return TestClient(app, base_url=base_url, client=("127.0.0.1", 50000))


def test_local_only_and_origin_boundary(monkeypatch):
    monkeypatch.delenv('APP_AUTH_TOKEN', raising=False)
    client = make_client()
    assert client.get('/').status_code == 200
    assert client.get('/', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert make_client('http://remote.example').get('/').status_code == 403


def test_network_basic_auth_and_websocket_session(monkeypatch):
    monkeypatch.setenv('APP_AUTH_TOKEN', 'network-secret')
    client = make_client('https://remote.example')
    assert client.get('/').status_code == 401
    assert client.get('/', auth=('theseus', 'wrong')).status_code == 401
    response = client.get('/', auth=('theseus', 'network-secret'))
    assert response.status_code == 200
    assert 'HttpOnly' in response.headers['set-cookie']
    assert client.get('/').status_code == 200
    with client.websocket_connect('wss://remote.example/ws') as socket:
        assert socket.receive_text() == 'ok'
    assert client.get('/', headers={'Origin': 'https://evil.example'}).status_code == 401
