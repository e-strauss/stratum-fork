from __future__ import annotations
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Hashable

from stratum._config import FLAGS
from stratum.runtime._object_serialization import (
    AtomicObject,
    delete_object,
    deserialize_object,
    serialize_object,
)
from stratum.runtime._object_size import get_size, prettify_bytes

logger = logging.getLogger(__name__)

_DEFAULT_SPILL_ROOT = Path(".stratum/bufferpool")


@dataclass(frozen=True)
class Handle:
    """A spilled buffer-pool entry.

    `structure` mirrors the original object: every list/tuple/dict is preserved
    in memory, and every atomic leaf is replaced by an `AtomicObject` reference
    pointing at the spilled file. Sizes aggregate over all leaves; the in-memory
    container shell is assumed negligible.
    """

    key: Hashable
    structure: Any
    size_on_disk: int
    size_in_memory: int


class Serializer:
    """Spill (possibly nested) objects to disk and reconstruct them on demand.

    Containers (list/tuple/dict) stay in memory; their atomic leaves are
    written to individual files under `root`. The root directory is created
    lazily on the first spill, so constructing a `Serializer` is a no-op on
    the filesystem.
    """

    def __init__(self, root: Path | str = _DEFAULT_SPILL_ROOT):
        self.root = Path(root)
        self._handles: dict[Hashable, Handle] = {}
        # Monotonic per-serialize counter so spill filenames are unique without
        # relying on `repr(key)` being injective for arbitrary Hashables.
        self._next_id: int = 0

    def serialize(self, key: Hashable, obj: Any) -> Handle:
        if key in self._handles:
            self.delete(key)
        self.root.mkdir(parents=True, exist_ok=True)
        prefix = f"{self._next_id:08x}"
        self._next_id += 1
        counter = [0]

        def spill(node: Any) -> Any:
            if isinstance(node, list):
                return [spill(x) for x in node]
            if isinstance(node, tuple):
                return tuple(spill(x) for x in node)
            if isinstance(node, dict):
                return {k: spill(v) for k, v in node.items()}
            stem = self.root / f"{prefix}_{counter[0]}"
            counter[0] += 1
            return serialize_object(node, stem)

        structure = spill(obj)
        leaves = list(_walk_leaves(structure))
        handle = Handle(
            key=key,
            structure=structure,
            size_on_disk=sum(l.size_on_disk for l in leaves),
            size_in_memory=sum(l.size_in_memory for l in leaves),
        )
        self._handles[key] = handle
        return handle

    def deserialize(self, key: Hashable) -> Any:
        return _rehydrate(self._handles[key].structure)

    def delete(self, key: Hashable) -> bool:
        handle = self._handles.pop(key, None)
        if handle is None:
            return False
        for leaf in _walk_leaves(handle.structure):
            delete_object(leaf)
        return True

    def clear(self) -> None:
        for key in list(self._handles.keys()):
            self.delete(key)


def _walk_leaves(node: Any):
    if isinstance(node, AtomicObject):
        yield node
    elif isinstance(node, (list, tuple)):
        for x in node:
            yield from _walk_leaves(x)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_leaves(v)


def _rehydrate(node: Any) -> Any:
    if isinstance(node, AtomicObject):
        return deserialize_object(node)
    if isinstance(node, list):
        return [_rehydrate(x) for x in node]
    if isinstance(node, tuple):
        return tuple(_rehydrate(x) for x in node)
    if isinstance(node, dict):
        return {k: _rehydrate(v) for k, v in node.items()}
    return node


@dataclass
class _Entry:
    """Internal record. `data` is None iff the entry has been spilled to disk."""
    size: int  # in-memory cost; constant after put
    data: Any = None
    handle: Handle | None = None
    pin_count: int = 0

    @property
    def is_spilled(self) -> bool:
        return self.handle is not None


@dataclass
class BufferPoolStats:
    """Counters/timers maintained over a BufferPool's lifetime.

    Hit = `pin` served the entry directly from memory.
    Miss = `pin` had to deserialize the entry back from disk.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    serialize_time: float = 0.0  # cumulative seconds spent spilling
    deserialize_time: float = 0.0  # cumulative seconds spent reloading
    bytes_spilled: int = 0
    bytes_loaded: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __str__(self) -> str:
        return (
            "BufferPool stats:\n"
            f"  Hits:             {self.hits}\n"
            f"  Misses:           {self.misses}\n"
            f"  Hit rate:         {self.hit_rate * 100:.1f}%\n"
            f"  Evictions:        {self.evictions}\n"
            f"  Serialize time:   {self.serialize_time:.3f}s\n"
            f"  Deserialize time: {self.deserialize_time:.3f}s\n"
            f"  Bytes spilled:    {prettify_bytes(self.bytes_spilled)}\n"
            f"  Bytes loaded:     {prettify_bytes(self.bytes_loaded)}"
        )


class BufferPool:
    """Cache for intermediate buffers with LRU spill-to-disk eviction.

    When `memory_usage` exceeds `memory_budget` (and budget > 0), the
    least-recently-pinned unpinned entries are spilled via `self.serializer`.
    A spilled entry stays in `live_variable_map` so subsequent `pin` calls can
    transparently re-load it; only the in-memory bytes are released.

    `memory_budget` is sourced from `FLAGS.buffer_pool_memory_budget` at
    construction time; a value of 0 disables eviction.
    """

    def __init__(self, serializer: Serializer | None = None):
        # TODO:
        # right now our buffer pool is basically a symbol table and (main memory) puffer pool in one thing
        # once we have multiple backends, we will have to separate both
        self.live_variable_map: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._removed_count: int = 0
        self.memory_usage = 0
        self.serializer = serializer if serializer is not None else Serializer()
        self.memory_budget = FLAGS.buffer_pool_memory_budget
        self.stats = BufferPoolStats()

    def put(self, key: Hashable, data: Any):
        """Store data for a key. Overwrites any existing entry."""
        if key in self.live_variable_map:
            self._drop(key)
        size = get_size(data)
        self.live_variable_map[key] = _Entry(size=size, data=data)
        self.memory_usage += size
        self.check_for_eviction()

    def pin(self, key: Hashable) -> Any:
        """Retrieve stored data for a key, or None if absent.

        Locks the entry against eviction (refcounted; balance every `pin` with
        an `unpin`). If the entry was previously spilled to disk, it is
        deserialized back into memory.
        """
        entry = self.live_variable_map.get(key)
        if entry is None:
            return None
        if entry.is_spilled:
            logger.debug(f"Deserializing {key} into memory: {prettify_bytes(entry.size)}")
            t0 = perf_counter()
            entry.data = self.serializer.deserialize(key)
            self.serializer.delete(key)
            self.stats.deserialize_time += perf_counter() - t0
            self.stats.misses += 1
            self.stats.bytes_loaded += entry.size
            entry.handle = None
            self.memory_usage += entry.size
        else:
            self.stats.hits += 1
        entry.pin_count += 1
        self.live_variable_map.move_to_end(key)
        return entry.data

    def unpin(self, key: Hashable) -> None:
        """Release a pinned buffer, allowing it to be evicted."""
        entry = self.live_variable_map.get(key)
        if entry is None:
            return
        if entry.pin_count > 0:
            entry.pin_count -= 1

    def remove(self, key: Hashable) -> bool:
        """Remove a single buffer, dropping its data (and any spilled file)."""
        if key not in self.live_variable_map:
            return False
        self._drop(key)
        self._removed_count += 1
        logger.debug(f"Removing buffer for {key}")
        assert self.memory_usage >= 0, "Memory usage is negative"
        return True

    def _drop(self, key: Hashable) -> None:
        """Internal: free in-memory bytes and any spilled file, then unlink the entry."""
        entry = self.live_variable_map.pop(key)
        if entry.is_spilled:
            self.serializer.delete(key)
        else:
            self.memory_usage -= entry.size

    def remove_all(self) -> list:
        """Remove everything, including pinned. Used at end of execution.

        Returns a list of removed keys.
        """
        removed = list(self.live_variable_map.keys())
        for key in removed:
            self.remove(key)
        return removed

    @property
    def active_count(self) -> int:
        return len(self.live_variable_map)

    @property
    def total_removed(self) -> int:
        return self._removed_count

    @property
    def total_size(self) -> str:
        """Pretty print the total size of the buffer pool."""
        return prettify_bytes(self.memory_usage)

    def check_for_eviction(self) -> bool:
        """Spill LRU unpinned entries until under budget. Returns True if any evicted."""
        if self.memory_budget <= 0 or self.memory_usage <= self.memory_budget:
            return False
        evicted = False
        # OrderedDict iterates oldest → newest; oldest = LRU.
        for key in list(self.live_variable_map.keys()):
            if self.memory_usage <= self.memory_budget:
                break
            entry = self.live_variable_map[key]
            if entry.pin_count > 0 or entry.is_spilled:
                continue
            self._evict(key, entry)
            evicted = True
        if self.memory_usage > self.memory_budget:
            logger.warning(
                f"Budget {prettify_bytes(self.memory_budget)} still exceeded after eviction "
                f"(usage {prettify_bytes(self.memory_usage)}); all remaining entries are pinned."
            )
        return evicted

    def _evict(self, key: Hashable, entry: _Entry) -> None:
        t0 = perf_counter()
        handle = self.serializer.serialize(key, entry.data)
        self.stats.serialize_time += perf_counter() - t0
        self.stats.evictions += 1
        self.stats.bytes_spilled += handle.size_on_disk
        self.memory_usage -= entry.size
        entry.data = None
        entry.handle = handle
        logger.debug(
            f"Evicted {key}: freed {prettify_bytes(entry.size)} "
            f"(spilled {prettify_bytes(handle.size_on_disk)} to disk)"
        )
