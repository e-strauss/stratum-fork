import argparse
import os
import sys
from time import perf_counter, sleep

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from utils.memory_consumption_tracker import MemoryTracker

import stratum as st
from sklearn.model_selection import ShuffleSplit
import pathlib

def plus_one(x, sleep_time=0):
    sleep(sleep_time)
    return x + 1
        

def row_sum(x, sleep_time=0):
    sleep(sleep_time)
    return x.sum(axis=1)

def multi_add(*args):
    i = 0
    for arg in args:
        i += arg
    return i

def pipe0(arr, y):
    arr2 = arr.skb.apply_func(plus_one, 0.30)
    arr3 = arr.skb.apply_func(plus_one, 0.31)
    arr4 = arr.skb.apply_func(plus_one, 0.32)
    arr5 = arr.skb.apply_func(plus_one, 0.33)
    tmp = arr2.skb.apply_func(multi_add, arr3, arr4, arr5)
    pred = tmp.skb.apply(DummyRegressor(), y=y)
    return pred

def fully_connected_binary_op_layer(nodes):
    new_layer = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            new_layer.append(nodes[i] + nodes[j])
    return new_layer

def pipe1(arr, y):
    # layer 1
    l1 = [arr.skb.apply_func(plus_one, 0.30),
          arr.skb.apply_func(plus_one, 0.31),
          arr.skb.apply_func(plus_one, 0.32),
          arr.skb.apply_func(plus_one, 0.33),
          arr.skb.apply_func(plus_one, 0.34),
          arr.skb.apply_func(plus_one, 0.35),
          ]

    # layer 2
    l2 = fully_connected_binary_op_layer(l1)

    # layer 3
    l3 = []
    offset = len(l2) // 2
    for i in range(offset):
        tmp = l2[i] + l2[i + offset]
        l3.append(tmp.skb.apply_func(row_sum, 0.36))

    final = l3[0].skb.apply_func(multi_add, *l3[1:])

    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred

def pipe2(arr, y):
    arr1_1 = arr.skb.apply_func(plus_one, 2.0)
    arr1_2 = arr1_1.skb.apply_func(plus_one, 2.0)
    arr1_3 = arr1_2.skb.apply_func(plus_one, 2.0)
    arr1_4 = arr1_3.skb.apply_func(plus_one, 2.0)

    arr2_2 = arr1_1.skb.apply_func(plus_one, 2.1)
    arr2_3 = arr2_2.skb.apply_func(plus_one, 2.1)
    arr2_4 = arr2_3.skb.apply_func(plus_one, 2.1)

    arr3_3 = arr1_2.skb.apply_func(plus_one, 2.2)
    arr3_4 = arr3_3.skb.apply_func(plus_one, 2.2)

    ar4_1 = arr.skb.apply_func(plus_one, 2.3)
    ar4_2 = ar4_1.skb.apply_func(plus_one, 2.3)
    ar4_3 = ar4_2.skb.apply_func(plus_one, 2.3)
    ar4_4 = ar4_3.skb.apply_func(plus_one, 2.3)

    final = arr1_4.skb.apply_func(multi_add, arr2_4, arr3_4, ar4_4)
    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred


def pipe3(arr, y):
    arr1_1 = arr.skb.apply_func(plus_one, 0.30)
    arr1_2 = arr1_1.skb.apply_func(plus_one, 0.30)
    arr1_3 = arr1_2.skb.apply_func(plus_one, 0.30)
    arr1_4 = arr1_3.skb.apply_func(plus_one, 0.30)

    pred = arr1_4.skb.apply(DummyRegressor(), y=y)
    return pred

def pipe4(arr, y):
    # layer 1
    n = 8
    l1 = [arr.skb.apply_func(plus_one, 0.3 + n*0.01) for _ in range(n)]

    # layer 2
    l2 = fully_connected_binary_op_layer(l1)

    # layer 3
    l3 = []
    offset = len(l2) // 2
    for i in range(offset):
        tmp = l2[i] + l2[i + offset]
        l3.append(tmp.skb.apply_func(row_sum, 0.36))

    final = l3[0].skb.apply_func(multi_add, *l3[1:])

    pred = final.skb.apply(DummyRegressor(), y=y)
    return pred

def main(use_skrub: bool, budget: int, n_rows: int, n_cols: int, pipe: int, draw: bool):
    # check if input exists
    x_path = pathlib.Path(f"input_x_{n_rows}_{n_cols}.npy")
    y_path = pathlib.Path(f"input_y_{n_rows}.npy")
    if not x_path.exists():
        np.save(x_path, np.random.random((n_rows, n_cols)).astype(np.float64), allow_pickle=False)
        np.save(y_path, np.random.random(n_rows).astype(np.float64), allow_pickle=False)

    with st.config_context(eager_data_ops=False):
        arr = st.as_data_op(x_path.resolve().as_posix()).skb.apply_func(np.load)
        y = st.as_data_op(y_path.resolve().as_posix()).skb.apply_func(np.load)
        y = y.skb.mark_as_y()
        arr = arr.skb.mark_as_X()
        if pipe == 0:
            pred = pipe0(arr, y)
        elif pipe == 1:
            pred = pipe1(arr, y)
        elif pipe == 2:
            pred = pipe2(arr, y)
        elif pipe == 3:
            pred = pipe3(arr, y)
        elif pipe == 4:
            pred = pipe4(arr, y)
        else:
            raise ValueError(f"Invalid pipe: {pipe}")

        if draw:
            pred.skb.draw_graph().open()
            exit()

        tracker = MemoryTracker(mode="process", interval_sec=0.02)
        tracker.start()
        t0 = perf_counter()
        try:
            cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            with st.config(scheduler=not use_skrub, buffer_pool_memory_budget=budget*1024*1024*1024, stats=True, DEBUG=True):
                search = pred.skb.make_grid_search(cv=cv, fitted=True, refit=False)
        finally:
            tracker.stop()
        t1 = perf_counter()

    csv_path = os.path.join(
        os.path.dirname(__file__),
        f"memory_usage_evictions_{'skrub' if use_skrub else 'stratum'}_pipe{pipe}_budget{budget}GB.csv",
    )
    tracker.write_csv(csv_path)

    print(f"Time taken: {t1 - t0:.2f}s")
    print(search.results_)
    plot_memory(csv_path)


def plot_memory(csv_path: str):
    data = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["time_sec"], data["rss_mb"], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSS (MB)")
    ax.set_title("Buffer Pool Evictions Benchmark - Memory Usage")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = csv_path.replace(".csv", ".pdf")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skrub", action="store_true")
    parser.add_argument("--budget", type=int, default=0, help="Memory budget in GB")
    parser.add_argument("--n-rows", type=int, default=30_000_000)
    parser.add_argument("--n-cols", type=int, default=100)
    parser.add_argument("--pipe", type=int, default=1, help="Pipeline to run")
    parser.add_argument("--draw", action="store_true", help="Draw the graph")
    args = parser.parse_args()
    main(
        use_skrub=args.skrub,
        budget=args.budget,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        pipe=args.pipe,
        draw=args.draw,
    )
