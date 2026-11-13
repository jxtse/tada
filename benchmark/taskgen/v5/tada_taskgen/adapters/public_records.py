"""Lazy public text access avoids opening an entire corpus to sample record IDs."""
from collections.abc import Mapping
from functools import lru_cache


class PublicRecords(Mapping):
    def __init__(self, keys, loader):
        self.keys_set = frozenset(keys)
        self.load = lru_cache(maxsize=64)(loader)

    def __iter__(self):
        return iter(sorted(self.keys_set))

    def __len__(self):
        return len(self.keys_set)

    def __getitem__(self, key):
        if key not in self.keys_set:
            raise KeyError(key)
        return self.load(key)
