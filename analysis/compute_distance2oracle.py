from __future__ import annotations

from pathlib import Path

from rich import print as printr

from seaborn import plotting_context
import seaborn as sb
import pandas as pd

from dacbench.logger import load_logs, log2dataframe
from dacbench.plotting import plot_performance_per_instance, plot_performance, plot

import matplotlib.pyplot as plt
import matplotlib

from instance_dac.utils.data_loading import load_generalization_data


sb.set_style("whitegrid")
sb.set_palette("colorblind")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def l1_dist(x: pd.Series) -> float:
    ret = None
    if "oracle" in x:
        ret = x["oracle"] - x["full"]
    return ret


def l2_dist(x: pd.Series) -> float:
    ret = None
    if "oracle" in x:
        ret = (x["oracle"] - x["full"]) ** 2
    return ret


if __name__ == "__main__":
    path = "../runs/Sigmoid"
    train_instance_set = "2D3M_train"
    path = Path(path) / train_instance_set

    path = Path(path)
    benchmark_name = path.parts[2]
    instance_set_id = path.parts[3]
    outdir = Path("figures") / benchmark_name / instance_set_id / "generalization"
    outdir.mkdir(exist_ok=True, parents=True)

    diffs = load_generalization_data(
        path=path, train_instance_set_id="sigmoid_2D3M_train", distance_functions=[l1_dist, l2_dist]
    )
    printr(diffs)
