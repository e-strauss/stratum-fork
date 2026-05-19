from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

from numpy import ndarray, load as np_load, save as np_save
from pandas import DataFrame, Series, read_parquet
from polars import DataFrame as PolarsDataFrame, Series as PolarsSeries, read_parquet as pl_read_parquet

from stratum.runtime._object_size import get_size

logger = getLogger(__name__)

_FORMAT_EXT = {
    "pandas_dataframe": ".parquet",
    "pandas_series": ".parquet",
    "polars_dataframe": ".parquet",
    "polars_series": ".parquet",
    "numpy_ndarray": ".npy",
    "pickle": ".pkl",
}


@dataclass(frozen=True)
class AtomicObject:
    """Reference to a single spilled atomic object."""
    path: Path
    format: str
    size_on_disk: int
    size_in_memory: int


def serialize_object(obj: Any, path_stem: Path) -> AtomicObject:
    """Spill a single atomic object. path_stem is the target path without extension."""
    fmt = _format_for(obj)
    path = Path(str(path_stem) + _FORMAT_EXT[fmt])
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write(obj, tmp, fmt)
    os.replace(tmp, path)
    return AtomicObject(path=path, format=fmt, size_on_disk=path.stat().st_size, size_in_memory=get_size(obj))


def deserialize_object(leaf: AtomicObject) -> Any:
    return _read(leaf.path, leaf.format)


def delete_object(leaf: AtomicObject) -> None:
    try:
        leaf.path.unlink()
    except FileNotFoundError:
        pass


def _format_for(obj: Any) -> str:
    if isinstance(obj, DataFrame):
        return "pandas_dataframe"
    if isinstance(obj, Series):
        return "pandas_series"
    if isinstance(obj, PolarsDataFrame):
        return "polars_dataframe"
    if isinstance(obj, PolarsSeries):
        return "polars_series"
    if isinstance(obj, ndarray):
        return "numpy_ndarray"
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        return "pickle"
    raise ValueError(f"Unsupported type for serialization: {type(obj)}")


def _write(obj: Any, path: Path, fmt: str) -> None:
    if fmt == "pandas_dataframe":
        obj.to_parquet(path)
    elif fmt == "pandas_series":
        obj.to_frame().to_parquet(path)
    elif fmt == "polars_dataframe":
        obj.write_parquet(path)
    elif fmt == "polars_series":
        obj.to_frame().write_parquet(path)
    elif fmt == "numpy_ndarray":
        with open(path, "wb") as f:
            np_save(f, obj, allow_pickle=False)
    elif fmt == "pickle":
        # Only used for our own spilled primitives; never load untrusted pickles.
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def _read(path: Path, fmt: str) -> Any:
    if fmt == "pandas_dataframe":
        return read_parquet(path)
    if fmt == "pandas_series":
        return read_parquet(path).iloc[:, 0]
    if fmt == "polars_dataframe":
        return pl_read_parquet(path)
    if fmt == "polars_series":
        return pl_read_parquet(path).to_series(0)
    if fmt == "numpy_ndarray":
        with open(path, "rb") as f:
            return np_load(f, allow_pickle=False)
    if fmt == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unknown format: {fmt}")
