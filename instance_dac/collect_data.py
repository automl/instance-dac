from pathlib import Path
from rich import print as printr
from omegaconf import OmegaConf
import pandas as pd
from dacbench.logger import Logger, log2dataframe, load_logs
import matplotlib.pyplot as plt
from dacbench.plotting import plot_performance, plot_performance_per_instance, plot_state
from multiprocessing import Pool
import os


cfg_fn = ".hydra/config.yaml"
perf_fn = "PerformanceTrackingWrapper.jsonl"
state_fn = "StateTrackingWrapper.jsonl"
reward_fn = "RewardTrackingWrapper.jsonl"
action_fn = "ActionFrequencyWrapper.jsonl"


def get_eval_df(eval_dir: Path) -> pd.DataFrame:
    # Get config
    cfg_filename = eval_dir / "../../.." / cfg_fn

    cfg = OmegaConf.load(cfg_filename)

    # Recover the correct test set path bc it gets overwritten
    cfg.benchmark.config.test_set_path = str(Path(cfg.benchmark.config.test_set_path).parent / (str(eval_dir.stem) + ".csv"))

    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)

    cfg_dict_flat = pd.json_normalize(data=cfg_dict, sep=".")

    cfg_small = {
        "benchmark_id": cfg.benchmark_id,
        "instance_set_id": cfg.instance_set_id,
        "test_set_id": Path(cfg.benchmark.config.test_set_path).name
    }

    # Read performance data
    logs = load_logs(eval_dir / perf_fn)
    perf_df = log2dataframe(logs, wide=True)

    # Read state data
    logs = load_logs(eval_dir / state_fn)
    state_df = log2dataframe(logs, wide=True)

    # Read reward data
    logs = load_logs(eval_dir / reward_fn)
    reward_df = log2dataframe(logs, wide=True)

    # Read action data
    logs = load_logs(eval_dir / action_fn)
    action_df = log2dataframe(logs, wide=True)

    index_columns = ["episode", "step", "seed", "instance"]

    # df = perf_df.merge(state_df)
    df = state_df.merge(reward_df)
    df = df.merge(action_df)

    for k, v in cfg_small.items():
        df[k] = v

    return df




if __name__ == "__main__":
    path = Path("runs2/Sigmoid")
    eval_dirs = list(path.glob("**/eval/*"))
    eval_dirs.sort()
    printr(eval_dirs)
    common_path = os.path.commonpath(eval_dirs)
    df_fn = Path("data") / common_path / "eval.csv"
    df_fn.parent.mkdir(parents=True, exist_ok=True)

    with Pool() as pool:
        dfs = pool.map(get_eval_df, eval_dirs)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(df_fn, index=False)
