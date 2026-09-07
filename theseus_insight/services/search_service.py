"""Bounded search execution and config-keyed embedding model reuse."""
from functools import lru_cache, wraps
import json
import logging
import os
import threading
import time
from fastapi import HTTPException

_model_lock = threading.Lock()
_slots = threading.BoundedSemaphore(max(1, int(os.getenv('SEARCH_CONCURRENCY', '2'))))


def bounded_search(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not _slots.acquire(blocking=False):
            raise HTTPException(429, 'Search capacity busy; retry shortly', headers={'Retry-After': '2'})
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            _slots.release()
            logging.getLogger(__name__).info('search endpoint=%s duration_ms=%.1f', fn.__name__, (time.monotonic()-started)*1000)
    return wrapped


@lru_cache(maxsize=2)
def _model(config_json):
    from ..inference import SentenceTransformerInference
    config = json.loads(config_json)
    return SentenceTransformerInference(config['model_name'], remote_code=config.get('trust_remote_code', False))


def embedding_model(config):
    with _model_lock:
        return _model(json.dumps(config, sort_keys=True))
