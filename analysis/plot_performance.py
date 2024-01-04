from __future__ import annotations

from pathlib import Path

from seaborn import plotting_context
import seaborn as sb

from dacbench.plotting import plot_performance_per_instance, plot_performance, plot

import matplotlib.pyplot as plt
import matplotlib

from instance_dac.utils.data_loading import load_performance_data


sb.set_style("whitegrid")
sb.set_palette("colorblind")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


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
    grid = plot_performance_per_instance(data, title="CMA Mean Performance per Instance")

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
        grid = plot_performance(data, title="Overall Performance", col="seed", col_wrap=3)
        grid.fig.subplots_adjust(top=0.92)
        grid.savefig(f"{outdir}/overall_performance_per_seed.pdf")
        plt.show()


def performance_example_over_time(path: str | Path):
    """
    Plot benchmark's performance over time, divided by seed and with each seed in its own plot
    """
    path = Path(path)
    benchmark_name = path.parts[2]
    instance_set_id = path.parts[3]
    outdir = Path("figures") / benchmark_name / instance_set_id
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_performance_data(path, drop_time=False)
    print(data)
    print(data.columns)

    # overall
    title = "Overall Performance"
    settings = {
        "data": data,
        "x": "time",
        "y": "overall_performance",
        "kind": "line",
    }
    x_label = y_label = None
    kwargs = {}
    grid = plot(sb.relplot, settings, title, x_label, y_label, **kwargs)
    grid.savefig(f"{outdir}/overall_performance_over_time.pdf")
    plt.show()


if __name__ == "__main__":
    path = "../runs/CMA-ES/default"

    # performance_example_over_time(path=path)
    per_instance_example(path=path)
    performance_example(path=path)
