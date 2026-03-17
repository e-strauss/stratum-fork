import argparse
import time
from contextlib import nullcontext
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from stratum.adapters.standard_scaler import NumpyStandardScaler, RustyStandardScaler
from stratum._config import config


def _run_single_case(n_rows: int, n_cols: int, seed: int = 0, rep: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)

    cases = (
        ("sklearn", SKStandardScaler(copy=True), nullcontext()),
        ("rust", RustyStandardScaler(copy=True), config(rust_backend=True, allow_patch=True, debug_timing=False)),
        ("numpy_copy_true", NumpyStandardScaler(copy=True), nullcontext()),
        ("numpy_copy_false", NumpyStandardScaler(copy=False), nullcontext()),
    )

    rows: list[dict] = []
    for backend, scaler, ctx in cases:
        with ctx:
            t0 = time.perf_counter()
            scaler.fit(X)
            t1 = time.perf_counter()
            fit_time = t1 - t0

            t0 = time.perf_counter()
            scaler.transform(X)
            t1 = time.perf_counter()
            transform_time = t1 - t0

        rows.append(
            {
                "backend": backend,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "fit_time_s": fit_time,
                "transform_time_s": transform_time,
                "rep": rep,
            }
        )
    return rows


def run_benchmark(
    n_rows_list: list[int],
    n_cols_list: list[int],
    *,
    reps: int = 3,
    seed: int = 0,
) -> pl.DataFrame:
    """Run the benchmark for all (n_rows, n_cols) combinations and return an averaged Polars DataFrame.

    Each (backend, n_rows, n_cols) combination is repeated `reps` times and
    timings are averaged over repetitions.
    """

    # init thread pool for rust
    _run_single_case(100, 10, seed=0, rep=1)

    all_rows: list[dict] = []
    for n_rows, n_cols in product(n_rows_list, n_cols_list):
        for rep in range(reps):
            # Change seed per repetition to avoid reusing the exact same data
            all_rows.extend(_run_single_case(n_rows* 1000, n_cols, seed=seed + rep, rep=rep))

    df = pl.DataFrame(all_rows)
    return (
        df.group_by(["backend", "n_rows", "n_cols"])
        .agg(
            pl.col("fit_time_s").mean().alias("fit_time_s"),
            pl.col("transform_time_s").mean().alias("transform_time_s"),
        )
        .sort(["backend", "n_rows", "n_cols"])
    )


def _plot_by_rows(
    df: pl.DataFrame,
    *,
    png: bool = False,
    ignore_sklearn: bool = False,
) -> None:
    """Plot fit+transform times grouped by n_rows, similar to plot_sc_by_rows."""
    n_rows_values = sorted(df["n_rows"].unique().to_list())
    backends = [
        "rust",
        "numpy_copy_true",
        "numpy_copy_false",
    ]
    if not ignore_sklearn:
        backends.append("sklearn")

    metric_colors = {
        "fit_time_s": "#1f77b4",
        "transform_time_s": "#2ca02c",
    }

    backend_hatches = {
        "rust": "///",
        "numpy_copy_true": "xxx",
        "numpy_copy_false": "...",
        "numba": "+++",
        "sklearn": "\\\\\\",
    }

    fig, axes = plt.subplots(
        nrows=len(n_rows_values),
        ncols=1,
        figsize=(6, 2.5 * len(n_rows_values)),
        sharex=True,
    )

    if len(n_rows_values) == 1:
        axes = [axes]

    bar_width = 0.26

    for ax, n_rows in zip(axes, n_rows_values):
        df_nr = df.filter(pl.col("n_rows") == n_rows)

        n_cols_values = sorted(df_nr["n_cols"].unique().to_list())
        group_width = bar_width * (len(backends) + 1)
        x = np.arange(len(n_cols_values)) * group_width

        for i, backend in enumerate(backends):
            df_b = (
                df_nr.filter(pl.col("backend") == backend)
                .sort("n_cols")
            )
            offset = (i - (len(backends) - 1) / 2) * bar_width

            fit = df_b["fit_time_s"]
            transform = df_b["transform_time_s"]

            ax.bar(
                x + offset,
                fit,
                width=bar_width,
                color=metric_colors["fit_time_s"],
                hatch=backend_hatches[backend],
                edgecolor="black",
            )
            ax.bar(
                x + offset,
                transform,
                width=bar_width,
                bottom=fit,
                color=metric_colors["transform_time_s"],
                hatch=backend_hatches[backend],
                edgecolor="black",
            )

        ax.set_title(f"n_rows = {n_rows}")
        ax.set_ylabel("time [s]")

    axes[-1].set_xticks(
        np.arange(len(n_cols_values)) * (bar_width * (len(backends) + 1)),
        [str(v) for v in n_cols_values],
    )
    axes[-1].set_xlabel("n_cols")

    metric_patches = [
        Patch(
            facecolor=metric_colors["fit_time_s"],
            edgecolor="black",
            label="fit",
        ),
        Patch(
            facecolor=metric_colors["transform_time_s"],
            edgecolor="black",
            label="transform",
        ),
    ]
    backend_patches = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=backend_hatches["rust"],
            label="rust",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=backend_hatches["numpy_copy_true"],
            label="numpy_copy_true",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=backend_hatches["numpy_copy_false"],
            label="numpy_copy_false",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=backend_hatches["sklearn"],
            label="sklearn",
        ),
    ]

    fig.legend(
        handles=metric_patches + backend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        title="metrics / backends",
    )

    fig.suptitle(
        "StandardScaler fit + transform times vs n_cols",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if png:
        plt.savefig("standard_scaler_benchmark_rows.png")
    else:
        plt.savefig("standard_scaler_benchmark_rows.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark StandardScaler implementations on random numpy arrays.\n"
            "Fit and transform times are measured separately for each combination\n"
            "of n_rows and n_cols."
        )
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        nargs="+",
        required=True,
        help="List of row counts to benchmark, in thousands, e.g. --n-rows 10 50",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        nargs="+",
        required=True,
        help="List of column counts to benchmark, e.g. --n-cols 10 100",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per (n_rows, n_cols) combination to average over (default: 3)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_false",
        dest="plot",
        help="Disable plotting of results grouped by rows (default: enabled)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="If plotting, save as PNG instead of PDF",
    )
    parser.add_argument(
        "--ignore_sklearn",
        action="store_true",
        help="If set, drop sklearn backend from plot",
    )

    args = parser.parse_args()

    df = run_benchmark(args.n_rows, args.n_cols, reps=args.reps, seed=0)
    output_path = "standard_scaler_benchmark.csv"
    df.write_csv(output_path)
    print(f"Wrote benchmark results to {output_path}")
    print(df.show(limit=df.shape[0]))

    if args.plot:
        _plot_by_rows(
            df,
            png=args.png,
            ignore_sklearn=args.ignore_sklearn,
        )


if __name__ == "__main__":
    main()

