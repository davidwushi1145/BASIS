"""
LRU Cache Implementation

Thread-safe Least Recently Used cache for gene aliases and other frequently accessed data.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    """
    Thread-safe LRU Cache.

    Features:
    - O(1) get and set operations
    - Thread-safe with lock
    - Automatic eviction of least recently used items
    """

    def __init__(self, capacity: int):
        """
        Initialize LRU Cache.

        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return default

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style assignment"""
        self.set(key, value)

    def set(self, key: str, value: Any):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to store
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.capacity:
                    # Evict least recently used (first item)
                    self.cache.popitem(last=False)

            self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        with self.lock:
            return key in self.cache

    def __len__(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()

    def keys(self):
        """Get all keys (snapshot)"""
        with self.lock:
            return list(self.cache.keys())

    def values(self):
        """Get all values (snapshot)"""
        with self.lock:
            return list(self.cache.values())
