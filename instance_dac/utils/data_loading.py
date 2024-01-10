from pathlib import Path

from rich import progress
import logging
from rich.logging import RichHandler
from rich.progress import Progress
from typing import Any
from rich import print as printr
from rich.progress import TimeElapsedColumn, MofNCompleteColumn
import numpy as np
from omegaconf import OmegaConf

from seaborn import plotting_context
import seaborn as sb
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from functools import partial

from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_performance_per_instance, plot_performance, plot


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


def _load_single_performance_data(filename: str, drop_time: bool = True) -> pd.DataFrame:
    logs = load_logs(filename)
    if drop_time:
        drop_columns = ["time"]
    else:
        drop_columns = None
    data = log2dataframe(logs, wide=True, drop_columns=drop_columns)

    if "eval" in str(filename):
        cfg_fn = Path(filename).parent.parent.parent.parent / ".hydra/config.yaml"
    else:
        cfg_fn = Path(filename).parent.parent.parent / ".hydra/config.yaml"
    cfg = OmegaConf.load(cfg_fn)
    data["instance_set_id"] = cfg.instance_set_id
    return data


def load_performance_data(
    path: str | Path, id: str = "PerformanceTrackingWrapper.jsonl", search_prefix: str = "", **kwargs
) -> pd.DataFrame:
    path = Path(path)
    filenames = list(path.glob(f"**/{search_prefix}{id}"))
    print(filenames)
    func = partial(_load_single_performance_data, **kwargs)
    dfs = map_multiprocessing(func, filenames)
    data = pd.concat(dfs).reset_index(drop=True)
    return data


def calc_dist(data: pd.Series, distance_function) -> pd.Series:
    ret = data.groupby("instance").apply(distance_function)
    ret.name = distance_function.__name__
    return ret


def load_generalization_data(
    path: str | Path, train_instance_set_id: str, distance_functions: list[callable]
) -> pd.DataFrame:
    path = Path(path)
    # Load full train set data, eval on train set
    data = load_performance_data(path, drop_time=True, search_prefix=f"full/**/eval/{train_instance_set_id}/")
    data["origin"] = "full"

    # Load oracle data
    oracle_data = load_performance_data(path, drop_time=True, search_prefix=f"oracle/**/eval/instance_*/")
    oracle_data["origin"] = "oracle"

    data = pd.concat([data, oracle_data])
    del oracle_data

    # Load selector data
    selector_data = load_performance_data(path, drop_time=True, search_prefix=f"selector/**/eval/{train_instance_set_id}/")
    selector_data["origin"] = "selector"

    data = pd.concat([data, oracle_data])
    del oracle_data

    # Aggregate performance per episode by mean, group by origin and instance
    perf = data.groupby(["origin", "instance"])["overall_performance"].mean()

    # Compute distance between oracle performance and performance on full training set
    diffs = pd.concat([calc_dist(perf, func) for func in distance_functions], axis=1).reset_index()
    diffs = diffs.melt(id_vars=["instance"], value_vars=[f.__name__ for f in distance_functions], value_name="distance", var_name="distance_name")

    return diffs
