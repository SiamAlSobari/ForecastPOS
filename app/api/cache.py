from datetime import datetime, timedelta
from hashlib import sha256
import json


class TTLCache:
    """Simple in-memory TTL cache for ML results."""

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[str, tuple[datetime, object]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, *args, **kwargs) -> str:
        raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return sha256(raw.encode()).hexdigest()

    def get(self, *args, **kwargs) -> object | None:
        key = self._make_key(*args, **kwargs)
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if datetime.now() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, value: object, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        self._store[key] = (datetime.now(), value)

    def clear(self):
        self._store.clear()


stock_cache = TTLCache(ttl_seconds=300)
busy_hour_cache = TTLCache(ttl_seconds=300)
insights_cache = TTLCache(ttl_seconds=300)
