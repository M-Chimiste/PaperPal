"""Versioned credential encryption and HTTP/WebSocket access control."""
import base64
import hashlib
import hmac
import ipaddress
import os
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from starlette.responses import JSONResponse

PREFIX = "fernet:v1:"


def cipher():
    secret = os.environ.get("APP_SECRET_KEY", "")
    if not secret or secret == "default_secret":
        raise ValueError("Set APP_SECRET_KEY before storing credentials")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


def encrypt(value):
    return PREFIX + cipher().encrypt(value.encode()).decode()


def decrypt(value):
    if not value.startswith(PREFIX):
        raise ValueError("Legacy credential requires explicit migration: scripts/migrate_credentials.py")
    return cipher().decrypt(value[len(PREFIX):].encode()).decode()


def migrate_legacy(value, encoding):
    """The old format was untagged; never guess plaintext versus ciphertext."""
    if value.startswith(PREFIX):
        decrypt(value)
        return value
    if encoding == "xor":
        secret = os.environ.get("APP_SECRET_KEY")
        if not secret:
            raise ValueError("APP_SECRET_KEY is required for legacy decryption")
        key = hashlib.sha256(secret.encode()).digest()
        data = base64.b64decode(value, validate=True)
        value = bytes(b ^ key[i % len(key)] for i, b in enumerate(data)).decode()
    elif encoding != "plaintext":
        raise ValueError("Choose plaintext or xor explicitly")
    return encrypt(value)


def loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AccessMiddleware:
    """Local-only by default; shared-token Basic auth for remote deployments.

    Browser Basic auth unlocks the app and issues a signed, HttpOnly session
    cookie for WebSockets. Tokens are never put in URLs or browser storage.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        host = urlsplit("//" + headers.get(b"host", b"").decode()).hostname
        origin = headers.get(b"origin", b"").decode()
        allowed = {x.strip().rstrip("/") for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()}
        origin_ok = not origin or origin in allowed or (
            urlsplit(origin).netloc == headers.get(b"host", b"").decode()
        ) or (loopback(host) and loopback(urlsplit(origin).hostname))
        token = os.getenv("APP_AUTH_TOKEN", "")
        peer = (scope.get("client") or ("", 0))[0]
        local = loopback(host) and loopback(peer)
        authenticated = not token and local
        basic_valid = False
        if token:
            auth = headers.get(b"authorization", b"").decode()
            if auth.startswith("Basic "):
                try:
                    password = base64.b64decode(auth[6:], validate=True).decode().split(":", 1)[1]
                    basic_valid = hmac.compare_digest(password, token)
                except (ValueError, IndexError, UnicodeError):
                    pass
            session_cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(token.encode()).digest()))
            from http.cookies import SimpleCookie
            cookie = SimpleCookie()
            try:
                cookie.load(headers.get(b"cookie", b"").decode())
                session = cookie.get("theseus_session")
                authenticated = bool(session and session_cipher.decrypt(session.value.encode(), ttl=43200) == b"session")
            except Exception:
                authenticated = False
            authenticated = authenticated or basic_valid
        # Liveness reveals no application state and is usable by container probes.
        probe = scope.get("path") in {"/health/live", "/health/ready"}
        if not probe and (not origin_ok or not authenticated):
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            response = JSONResponse({"detail": "Authentication required" if token else "Local access only; configure APP_AUTH_TOKEN for remote access"},
                                    status_code=401 if token else 403,
                                    headers={"WWW-Authenticate": 'Basic realm="Theseus Insight"'} if token else {})
            return await response(scope, receive, send)

        async def send_session(message):
            if basic_valid and message["type"] == "http.response.start":
                value = session_cipher.encrypt(b"session").decode()
                secure = "; Secure" if scope.get("scheme") == "https" else ""
                message.setdefault("headers", []).append((b"set-cookie", f"theseus_session={value}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200{secure}".encode()))
            await send(message)
        await self.app(scope, receive, send_session)
