"""Small per-instance LRU caches; no global references to fitted codecs."""

from collections import OrderedDict
from collections.abc import MutableMapping
from threading import RLock


def cached_prefix_state(key, cache, initial, advance):
    """Extend a cached parent prefix, replaying only when it was evicted."""
    if not key:
        return initial
    missing = object()
    state = cache.get(key, missing)
    if state is not missing:
        return state
    parent = key[:-1]
    state = cache.get(parent, missing)
    if state is missing:
        state = initial
        for token in parent:
            state = advance(state, token)
            if state is None:
                break
        cache[parent] = state
    if state is not None:
        state = advance(state, key[-1])
    cache[key] = state
    return state


class BoundedCache(MutableMapping):
    """Evict least-recently-used entries above a fixed capacity."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self.capacity = capacity
        self._items = OrderedDict()
        self._lock = RLock()

    def __getitem__(self, key):
        with self._lock:
            value = self._items[key]
            self._items.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def __delitem__(self, key):
        with self._lock:
            del self._items[key]

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._items))

    def __len__(self):
        with self._lock:
            return len(self._items)

    def reserve(self, capacity):
        """Raise the capacity so callers with a known working set stop thrashing."""
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        with self._lock:
            self.capacity = max(self.capacity, capacity)
