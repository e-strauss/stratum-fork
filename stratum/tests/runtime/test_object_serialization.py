import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from polars.testing import assert_series_equal as pl_assert_series_equal

from stratum.runtime._object_serialization import (
    AtomicObject,
    delete_object,
    deserialize_object,
    serialize_object,
)
from stratum.runtime._object_size import get_size


class TestLeafIO(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _stem(self, name: str = "x") -> Path:
        return self.root / name

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        leaf = serialize_object(df, self._stem())
        pd.testing.assert_frame_equal(deserialize_object(leaf), df)
        self.assertEqual(leaf.format, "pandas_dataframe")
        self.assertEqual(leaf.size_in_memory, get_size(df))
        self.assertGreater(leaf.size_on_disk, 0)

    def test_pandas_series(self):
        ser = pd.Series([1, 2, 3], name="x")
        leaf = serialize_object(ser, self._stem())
        pd.testing.assert_series_equal(deserialize_object(leaf), ser)

    def test_polars_dataframe(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        leaf = serialize_object(df, self._stem())
        pl_assert_frame_equal(deserialize_object(leaf), df)

    def test_polars_series(self):
        ser = pl.Series("x", [1, 2, 3])
        leaf = serialize_object(ser, self._stem())
        pl_assert_series_equal(deserialize_object(leaf), ser)

    def test_numpy(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        leaf = serialize_object(arr, self._stem())
        np.testing.assert_array_equal(deserialize_object(leaf), arr)
        self.assertEqual(leaf.size_in_memory, arr.nbytes)

    def test_primitives(self):
        for i, val in enumerate(["hello", 42, 3.14, True, b"raw", None]):
            leaf = serialize_object(val, self._stem(f"p{i}"))
            self.assertEqual(deserialize_object(leaf), val)
            self.assertEqual(leaf.format, "pickle")

    def test_unsupported_type_raises(self):
        class Foo:
            pass

        with self.assertRaises(ValueError):
            serialize_object(Foo(), self._stem())

    def test_no_tmp_after_serialize(self):
        serialize_object(np.arange(10), self._stem())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_delete_leaf(self):
        leaf = serialize_object("hello", self._stem())
        self.assertTrue(leaf.path.exists())
        delete_object(leaf)
        self.assertFalse(leaf.path.exists())
        delete_object(leaf)  # idempotent


if __name__ == "__main__":
    unittest.main()
