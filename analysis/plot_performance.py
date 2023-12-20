from pathlib import Path

from rich import progress
import logging
from rich.logging import RichHandler
from rich.progress import Progress
from typing import Any
from rich.progress import TimeElapsedColumn, MofNCompleteColumn

from seaborn import plotting_context
import seaborn as sb
import pandas as pd
import multiprocessing
from multiprocessing import Pool

from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_performance_per_instance, plot_performance

import matplotlib.pyplot as plt

sb.set_style("whitegrid")
sb.set_palette("colorblind")


def map_multiprocessing(
    task_function: callable, task_params: list[Any], n_processes: int = 4, task_string: str = "Working..."
) -> list:
    results = []
    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        refresh_per_second=2,
    ) as progress:
        task_id = progress.add_task(f"[cyan]{task_string}", total=len(task_params))
        with multiprocessing.Pool(processes=n_processes) as pool:
            for result in pool.imap(task_function, task_params):
                results.append(result)
                progress.advance(task_id)
    return results


def _load_single_performance_data(filename: str) -> pd.DataFrame:
    logs = load_logs(filename)
    data = log2dataframe(logs, wide=True, drop_columns=["time"])
    return data

def load_performance_data(path: str | Path, id: str = "PerformanceTrackingWrapper.jsonl") -> pd.DataFrame:
    path = Path(path)
    filenames = list(path.glob(f"**/{id}"))
    dfs = map_multiprocessing(_load_single_performance_data, filenames)
    data = pd.concat(dfs).reset_index(drop=True)
    return data


def per_instance_example(path: str | Path):
    """
    Plot CMA performance for each training instance
    """
    path = Path(path)
    benchmark_name = path.parts[2] 
    instance_set_id = path.parts[3]
    outdir = Path("figures") / benchmark_name / instance_set_id
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_performance_data(path)
    grid = plot_performance_per_instance(
        data, title="CMA Mean Performance per Instance"
    )

    grid.savefig(f"{outdir}/performance_per_instance.pdf")
    plt.show()

def performance_example(path: str | Path):
    """
    Plot benchmark's performance over time, divided by seed and with each seed in its own plot
    """
    path = Path(path)
    benchmark_name = path.parts[2] 
    instance_set_id = path.parts[3]
    outdir = Path("figures") / benchmark_name / instance_set_id
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_performance_data(path)
    Path("figures").mkdir(exist_ok=True)

    # overall
    grid = plot_performance(data, title="Overall Performance")
    grid.savefig(f"{outdir}/overall_performance.pdf")
    plt.show()

    # per instance seed (hue)
    grid = plot_performance(data, title="Overall Performance", hue="seed", marker=".")
    grid.savefig(f"{outdir}/overall_performance_per_seed_hue.pdf")
    plt.show()

    # per instance seed (col)
    with plotting_context("poster"):
        grid = plot_performance(
            data, title="Overall Performance", col="seed", col_wrap=3
        )
        grid.fig.subplots_adjust(top=0.92)
        grid.savefig(f"{outdir}/overall_performance_per_seed.pdf")
        plt.show()


if __name__ == "__main__":
    path = "../runs/CMA-ES/default"
    
    per_instance_example(path=path)
    performance_example(path=path)
