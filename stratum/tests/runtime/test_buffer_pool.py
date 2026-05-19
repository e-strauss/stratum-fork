import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from stratum._api import grid_search
from stratum._config import config
from stratum.runtime._buffer_pool import BufferPool, Handle, Serializer
from stratum.runtime._object_serialization import AtomicObject
from stratum.runtime._object_size import get_size
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, _arr, _make_op, simple_pipeline
import skrub
from sklearn.dummy import DummyRegressor


class TestBufferPool(unittest.TestCase):
    """Tests for BufferPool as a pure cache."""

    def test_put_and_get(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "data_x")
        self.assertEqual(pool.pin(op), "data_x")
        self.assertEqual(pool.active_count, 1)

    def test_get_missing_returns_none(self):
        pool = BufferPool()
        self.assertIsNone(pool.pin(_make_op("missing")))

    def test_remove_drops_data(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "data_x")
        removed = pool.remove(op)
        self.assertTrue(removed)
        self.assertIsNone(pool.pin(op))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_removed, 1)

    def test_remove_missing_returns_false(self):
        pool = BufferPool()
        self.assertFalse(pool.remove(_make_op("missing")))

    def test_remove_all(self):
        pool = BufferPool()
        ops = [_make_op(f"op{i}") for i in range(3)]
        for i, op in enumerate(ops):
            pool.put(op, f"data_{i}")

        removed = pool.remove_all()
        self.assertEqual(set(removed), set(ops))
        self.assertEqual(pool.active_count, 0)
        self.assertEqual(pool.total_removed, 3)

    def test_put_overwrites_existing(self):
        pool = BufferPool()
        op = _make_op("x")
        pool.put(op, "old_data")
        pool.put(op, "new_data")
        self.assertEqual(pool.pin(op), "new_data")
        self.assertEqual(pool.active_count, 1)

    def test_memory_usage(self):
        pool = BufferPool()
        op = _make_op("x")
        data_x = np.random.random(1024).astype(np.float64)
        pool.put(op, data_x)
        self.assertEqual(pool.memory_usage, 1024*8)
        pool.remove(op)
        self.assertEqual(pool.memory_usage, 0)
        pool.memory_usage = 2*1024**5
        self.assertEqual(pool.total_size, "2.00 PB")

    def test_unknown_object_sizes(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            get_size(Foo())

        with self.assertRaises(ValueError):
            get_size(pd.Index([1, 2, 3]))

        with self.assertRaises(ValueError):
            get_size(pl.LazyFrame({"a": [1, 2, 3]}))

        with self.assertRaises(ValueError):
            get_size(np.dtype("float64"))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestBufferPoolIntegration(RuntimeTest):

    def test_evaluate_matches_baseline(self):
        """Buffer-managed evaluate produces same results as skrub baseline."""
        pred_opt = simple_pipeline()
        self.compare_evaluate(pred_opt)

    def test_grid_search_runs(self):
        """Grid search with buffer manager completes without error."""
        pred_opt = simple_pipeline()
        results = grid_search(pred_opt, cv=2)
        self.assertIsNotNone(results)
        
    def test_buffer_pool_evictions(self):
        arr = skrub.as_data_op(_arr(1000)).skb.mark_as_X()
        y = skrub.as_data_op(_arr(1000)).skb.mark_as_y()
        def dummy_op(x):
            return x + 1

        arr2 = arr.skb.apply_func(lambda x: dummy_op(x)).reshape(-1, 1)
        arr3 = arr.skb.apply_func(lambda x: dummy_op(x)).reshape(-1, 1)
        arr4 = arr.skb.apply_func(lambda x: dummy_op(x)).reshape(-1, 1)
        arr_out = arr2.skb.apply_func(lambda a, b, c: np.hstack([a, b, c]), arr3, arr4)
        pred = arr_out.skb.apply(DummyRegressor(), y=y)

        with self.assertLogs("stratum", level="DEBUG") as logs:
            with config(scheduler=True, buffer_pool_memory_budget=24000, DEBUG=True):
                search = pred.skb.make_grid_search(cv=2)
        self.assertIsNotNone(search)
        evictions = [line for line in logs.output if "Evicted" in line]
        self.assertEqual(len(evictions), 12, msg="\n".join(logs.output))


class TestSerializer(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.s = Serializer(root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_atomic_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        h = self.s.serialize("k", df)
        self.assertIsInstance(h.structure, AtomicObject)
        pd.testing.assert_frame_equal(self.s.deserialize("k"), df)
        self.assertEqual(h.size_in_memory, get_size(df))

    def test_nested_tuple_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        arr = np.arange(5, dtype=np.float64)
        h = self.s.serialize("k", (df, arr))
        self.assertIsInstance(h.structure, tuple)
        self.assertEqual(len(h.structure), 2)
        self.assertTrue(all(isinstance(x, AtomicObject) for x in h.structure))
        out_df, out_arr = self.s.deserialize("k")
        pd.testing.assert_frame_equal(out_df, df)
        np.testing.assert_array_equal(out_arr, arr)
        self.assertEqual(h.size_on_disk, sum(x.size_on_disk for x in h.structure))

    def test_nested_dict_with_list_roundtrip(self):
        obj = {
            "df": pd.DataFrame({"a": [1, 2]}),
            "arrs": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "label": "train",
        }
        self.s.serialize("k", obj)
        out = self.s.deserialize("k")
        pd.testing.assert_frame_equal(out["df"], obj["df"])
        for a, b in zip(out["arrs"], obj["arrs"]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(out["label"], "train")

    def test_unsupported_leaf_raises(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            self.s.serialize("k", [pd.DataFrame({"a": [1]}), Foo()])

    def test_delete_removes_all_leaf_files(self):
        self.s.serialize("k", (np.arange(3), np.arange(4)))
        leaf_paths = [l.path for l in self.s._handles["k"].structure]
        self.assertTrue(all(p.exists() for p in leaf_paths))
        self.assertTrue(self.s.delete("k"))
        self.assertFalse(any(p.exists() for p in leaf_paths))
        self.assertNotIn("k", self.s._handles)
        self.assertFalse(self.s.delete("k"))

    def test_clear(self):
        self.s.serialize("a", "x")
        self.s.serialize("b", [np.arange(3), "y"])
        self.s.clear()
        self.assertEqual(len(self.s._handles), 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_overwrite_replaces_old_leaves(self):
        self.s.serialize("k", [np.arange(3), np.arange(4), np.arange(5)])
        self.assertEqual(len(list(self.root.iterdir())), 3)
        self.s.serialize("k", np.arange(2))
        self.assertEqual(len(list(self.root.iterdir())), 1)
        np.testing.assert_array_equal(self.s.deserialize("k"), np.arange(2))

    def test_handle_size_aggregation(self):
        h = self.s.serialize("k", [np.arange(10), np.arange(20)])
        leaf_disk = sum(l.size_on_disk for l in h.structure)
        leaf_mem = sum(l.size_in_memory for l in h.structure)
        self.assertEqual(h.size_on_disk, leaf_disk)
        self.assertEqual(h.size_in_memory, leaf_mem)


class TestLRUEviction(unittest.TestCase):
    """LRU spill-to-disk eviction in BufferPool."""

    def setUp(self):
        self.spill_root = Path(tempfile.mkdtemp())
        self.pool = BufferPool(serializer=Serializer(root=self.spill_root))
        self.pool.memory_budget = 10_000  # ~10 KB

    def tearDown(self):
        shutil.rmtree(self.spill_root, ignore_errors=True)

    def test_no_eviction_when_budget_zero(self):
        self.pool.memory_budget = 0  # override the setUp budget for this test
        a, b = _make_op("a"), _make_op("b")
        self.pool.put(a, _arr(1000))
        self.pool.put(b, _arr(1000))
        self.assertEqual(list(self.spill_root.iterdir()), [])
        self.assertEqual(self.pool.memory_usage, 16_000)

    def test_eviction_spills_lru(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        self.pool.put(a, _arr(500))  # 4_000 B; touches a
        self.pool.put(b, _arr(500))  # 4_000 B; touches b
        self.pool.put(c, _arr(500))  # 4_000 B → 12_000 total, over budget
        # `a` is LRU → should be the one spilled.
        self.assertTrue(self.pool.live_variable_map[a].is_spilled)
        self.assertFalse(self.pool.live_variable_map[b].is_spilled)
        self.assertFalse(self.pool.live_variable_map[c].is_spilled)
        self.assertEqual(self.pool.memory_usage, 8_000)

    def test_pin_reloads_evicted_entry(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        original = _arr(500)
        self.pool.put(a, original)
        self.pool.put(b, _arr(500))
        self.pool.put(c, _arr(500))  # evicts a
        self.assertTrue(self.pool.live_variable_map[a].is_spilled)

        out = self.pool.pin(a)
        np.testing.assert_array_equal(out, original)
        self.assertFalse(self.pool.live_variable_map[a].is_spilled)

    def test_pinned_entries_not_evicted(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        self.pool.put(a, _arr(500))
        self.pool.pin(a)  # protect a
        self.pool.put(b, _arr(500))
        self.pool.put(c, _arr(500))  # would normally evict a (LRU)
        self.assertFalse(self.pool.live_variable_map[a].is_spilled)
        # b is next LRU among unpinned, so b should have been spilled instead.
        self.assertTrue(self.pool.live_variable_map[b].is_spilled)

    def test_pin_refcount(self):
        a = _make_op("a")
        self.pool.put(a, "data")
        self.pool.pin(a)
        self.pool.pin(a)
        self.assertEqual(self.pool.live_variable_map[a].pin_count, 2)
        self.pool.unpin(a)
        self.assertEqual(self.pool.live_variable_map[a].pin_count, 1)
        self.pool.unpin(a)
        self.assertEqual(self.pool.live_variable_map[a].pin_count, 0)
        self.pool.unpin(a)  # extra unpin clamps at 0
        self.assertEqual(self.pool.live_variable_map[a].pin_count, 0)

    def test_pin_moves_to_mru(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        self.pool.put(a, _arr(500))
        self.pool.put(b, _arr(500))
        self.pool.pin(a)  # makes a the MRU
        self.pool.unpin(a)
        self.pool.put(c, _arr(500))  # over budget → evict LRU = b now, not a
        self.assertFalse(self.pool.live_variable_map[a].is_spilled)
        self.assertTrue(self.pool.live_variable_map[b].is_spilled)

    def test_remove_evicted_cleans_disk(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        self.pool.put(a, _arr(500))
        self.pool.put(b, _arr(500))
        self.pool.put(c, _arr(500))
        self.assertTrue(self.pool.live_variable_map[a].is_spilled)
        self.assertEqual(len(list(self.spill_root.iterdir())), 1)
        self.assertTrue(self.pool.remove(a))
        self.assertEqual(list(self.spill_root.iterdir()), [])

    def test_remove_all_cleans_disk(self):
        self.pool.memory_budget = 10_000
        for name in ("a", "b", "c"):
            self.pool.put(_make_op(name), _arr(500))
        self.assertGreater(len(list(self.spill_root.iterdir())), 0)
        self.pool.remove_all()
        self.assertEqual(self.pool.memory_usage, 0)
        self.assertEqual(list(self.spill_root.iterdir()), [])

    def test_put_overwrites_evicted(self):
        a, b, c = _make_op("a"), _make_op("b"), _make_op("c")
        self.pool.put(a, _arr(500))
        self.pool.put(b, _arr(500))
        self.pool.put(c, _arr(500))  # a now spilled
        self.assertTrue(self.pool.live_variable_map[a].is_spilled)
        self.pool.put(a, _arr(100))  # overwrite spilled entry
        self.assertFalse(self.pool.live_variable_map[a].is_spilled)
        self.assertEqual(self.pool.live_variable_map[a].size, 800)


if __name__ == "__main__":
    unittest.main()
