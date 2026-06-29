import unittest

import numpy as np
import pandas as pd
import polars as pl
import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._source_ops import DataSourceOp, make_read_op
from stratum.optimizer.ir._ops import CallOp, OperandRef, ValueOp
from stratum.runtime._buffer_pool import BufferPool
from stratum.tests.logical_optimizer.test_dataframe_ops import (
    csv_file, force_polars, npy_file, optimize, parquet_file)


class TestDataSourceRewrites(unittest.TestCase):
    """`optimize` turns a directly-passed frame / a read call into a DataSourceOp."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_data_source_from_dataframe(self):
        ops = optimize(st.as_data_op(self.df))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], DataSourceOp)

    def test_data_source_from_read_csv(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], DataSourceOp)

    def test_data_source_from_np_load(self):
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, DataSourceOp) and op.format == "npy"
                            for op in ops))

    def test_data_source_from_read_parquet(self):
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, DataSourceOp) and op.format == "parquet"
                            for op in ops))


class TestDataSourceOp(unittest.TestCase):
    def test_unsupported_format_raises(self):
        op = DataSourceOp(file_path="nofile", _format="orc",
                          read_args=(), read_kwargs={})
        with self.assertRaises(ValueError):
            op.process("fit_transform", [])

    def test_numpy_read(self):
        with npy_file(np.array([1, 2, 3])) as path:
            op = DataSourceOp(file_path=path, _format="npy",
                              read_args=(), read_kwargs={})
            result = op.process("fit_transform", [])
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_polars_from_dataframe(self):
        with force_polars():
            op = DataSourceOp(data=pd.DataFrame({"a": [1, 2]}))
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_polars_from_read_csv(self):
        with csv_file(pd.DataFrame({"a": [1, 2]})) as path, force_polars():
            op = DataSourceOp(file_path=path, _format="csv",
                              read_args=(), read_kwargs={})
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_pandas_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) as path:
            op = DataSourceOp(file_path=path, _format="parquet",
                              read_args=(), read_kwargs={})
            result = op.process("fit_transform", [])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual([1, 2], result["a"].tolist())

    def test_polars_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2]})) as path, force_polars():
            op = DataSourceOp(file_path=path, _format="parquet",
                              read_args=(), read_kwargs={})
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)


class TestMakeReadOp(unittest.TestCase):
    """`make_read_op` and its end-to-end usage via the optimizer."""

    def _optimize_read(self, data, env=None):
        with st.config(fast_dataops_convert=True):
            return optimize(data, OptConfig(dataframe_ops=True), env=env)

    def test_with_variable_input(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.var("path").skb.apply_func(pd.read_csv)
            # Resolve the path variable at compile time so the plan runs with no env.
            ops = self._optimize_read(data, env={"path": path})
            self.assertIsInstance(ops[-1], DataSourceOp)

            # Verify the resulting plan actually runs without a runtime environment.
            pool = BufferPool()
            for op in ops:
                inputs = [pool.pin(key) for key in op.inputs]
                pool.put(op, op.process("fit_transform", inputs))
            self.assertIsInstance(pool.pin(ops[-1]), pd.DataFrame)

    def test_with_variable_kwarg(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=st.var("path"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)

    def test_with_plain_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=";")
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_dataop_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(
                pd.read_csv, sep=st.as_data_op(";"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_plain_positional_arg(self):
        call_op = CallOp(func=pd.read_csv,
                         args=(OperandRef(0), ","), kwargs={})
        call_op.inputs = [ValueOp("dummy.csv")]
        new_op = make_read_op(call_op)
        self.assertIsInstance(new_op, DataSourceOp)
        self.assertEqual((",",), tuple(new_op.read_args))


if __name__ == "__main__":
    unittest.main()
