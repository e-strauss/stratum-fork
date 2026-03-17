import argparse
import time
from itertools import product

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from stratum.adapters.standard_scaler import RustyStandardScaler
from stratum._config import config


def _run_single_case(
    n_rows: int,
    n_cols: int,
    n_jobs: int,
    *,
    seed: int = 0,
    rep: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)

    scaler = RustyStandardScaler(copy=True, n_jobs=n_jobs)
    ctx = config(rust_backend=True, allow_patch=True, debug_timing=False)

    rows: list[dict] = []
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
            "backend": "rust",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_jobs": n_jobs,
            "fit_time_s": fit_time,
            "transform_time_s": transform_time,
            "rep": rep,
        }
    )
    return rows


def run_benchmark(
    n_rows_list: list[int],
    n_cols_list: list[int],
    n_jobs_list: list[int],
    *,
    reps: int = 3,
    seed: int = 0,
) -> pl.DataFrame:
    """Benchmark RustyStandardScaler for different n_jobs, rows and cols."""

    # warm up Rust thread pool / JIT, etc.
    _ = _run_single_case(100, 10, n_jobs=max(n_jobs_list), seed=seed, rep=0)

    all_rows: list[dict] = []
    for n_rows, n_cols, n_jobs in product(n_rows_list, n_cols_list, n_jobs_list):
        for rep in range(reps):
            # as in bench_standard_scaler1, interpret n_rows as thousands
            all_rows.extend(
                _run_single_case(
                    n_rows * 1000,
                    n_cols,
                    n_jobs,
                    seed=seed + rep,
                    rep=rep,
                )
            )

    df = pl.DataFrame(all_rows)
    return (
        df.group_by(["backend", "n_rows", "n_cols", "n_jobs"])
        .agg(
            pl.col("fit_time_s").mean().alias("fit_time_s"),
            pl.col("transform_time_s").mean().alias("transform_time_s"),
        )
        .sort(["backend", "n_rows", "n_cols", "n_jobs"])
    )


def _plot_by_rows(df: pl.DataFrame, *, png: bool = False) -> None:
    """Plot fit+transform times grouped by n_rows, similar to plot_sc_by_rows."""
    n_rows_values = sorted(df["n_rows"].unique().to_list())
    n_jobs_values = sorted(df["n_jobs"].unique().to_list())

    metric_colors = {
        "fit_time_s": "#1f77b4",
        "transform_time_s": "#2ca02c",
    }

    # Use different hatches per n_jobs
    hatch_patterns = ["///", "\\\\\\", "xxx", "...", "+++", "|||", "ooo"]
    n_hatches = len(hatch_patterns)
    n_jobs_hatches = {
        n_jobs: hatch_patterns[i % n_hatches]
        for i, n_jobs in enumerate(n_jobs_values)
    }

    fig, axes = plt.subplots(
        nrows=len(n_rows_values),
        ncols=1,
        figsize=(6, 2.5 * len(n_rows_values)),
        sharex=True,
    )

    if len(n_rows_values) == 1:
        axes = [axes]

    bar_width = 0.22
    group_width = bar_width * (len(n_jobs_values) + 1)

    for ax, n_rows in zip(axes, n_rows_values):
        df_nr = (
            df.filter(pl.col("n_rows") == n_rows)
            .filter(pl.col("backend") == "rust")
        )

        # Base positions per n_cols
        n_cols_values = sorted(df_nr["n_cols"].unique().to_list())
        x_base = np.arange(len(n_cols_values)) * group_width

        for j, n_jobs in enumerate(n_jobs_values):
            df_nj = (
                df_nr.filter(pl.col("n_jobs") == n_jobs)
                .sort("n_cols")
            )
            # In case some (n_cols, n_jobs) combos are missing, align by n_cols
            df_nj = df_nj.join(
                pl.DataFrame({"n_cols": n_cols_values}),
                on="n_cols",
                how="right",
            ).fill_null(strategy="zero")

            offset = (j - (len(n_jobs_values) - 1) / 2) * bar_width
            x = x_base + offset

            fit = df_nj["fit_time_s"]
            transform = df_nj["transform_time_s"]

            ax.bar(
                x,
                fit,
                width=bar_width,
                color=metric_colors["fit_time_s"],
                hatch=n_jobs_hatches[n_jobs],
                edgecolor="black",
            )
            ax.bar(
                x,
                transform,
                width=bar_width,
                bottom=fit,
                color=metric_colors["transform_time_s"],
                hatch=n_jobs_hatches[n_jobs],
                edgecolor="black",
            )

        ax.set_title(f"n_rows = {n_rows}")
        ax.set_ylabel("time [s]")
        ax.set_xticks(x_base, [str(v) for v in n_cols_values])

    axes[-1].set_xlabel("n_cols")

    metric_patches = [
        Patch(facecolor=metric_colors["fit_time_s"], edgecolor="black", label="fit"),
        Patch(facecolor=metric_colors["transform_time_s"], edgecolor="black", label="transform"),
    ]
    n_jobs_patches = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=n_jobs_hatches[n_jobs],
            label=f"n_jobs={n_jobs}",
        )
        for n_jobs in n_jobs_values
    ]

    fig.legend(
        handles=metric_patches + n_jobs_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        title="metrics / backends",
    )

    fig.suptitle("Rust StandardScaler fit + transform times vs n_cols", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if png:
        plt.savefig("standard_scaler_n_jobs_benchmark_rows.png")
    else:
        plt.savefig("standard_scaler_n_jobs_benchmark_rows.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RustyStandardScaler n_jobs on random numpy arrays.\n"
            "Fit and transform times are measured separately for each combination\n"
            "of n_rows, n_cols and n_jobs."
        )
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        nargs="+",
        required=True,
        help="List of row counts to benchmark, in thousands, e.g. --n_rows 10 50",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        nargs="+",
        required=True,
        help="List of column counts to benchmark, e.g. --n_cols 10 100",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        nargs="+",
        required=True,
        help="List of n_jobs values for RustyStandardScaler, e.g. --n_jobs 1 2 4 8",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per (n_rows, n_cols, n_jobs) combination to average over (default: 3)",
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

    args = parser.parse_args()

    df = run_benchmark(
        args.n_rows,
        args.n_cols,
        args.n_jobs,
        reps=args.reps,
        seed=0,
    )
    output_path = "standard_scaler_n_jobs_benchmark.csv"
    df.write_csv(output_path)
    print(f"Wrote benchmark results to {output_path}")
    print(df.show(limit=df.shape[0]))

    if args.plot:
        _plot_by_rows(df, png=args.png)


if __name__ == "__main__":
    main()

