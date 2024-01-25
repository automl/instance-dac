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
    cfg.benchmark.config.test_set_path = str(
        Path(cfg.benchmark.config.test_set_path).parent / (str(eval_dir.stem) + ".csv")
    )

    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)

    cfg_dict_flat = pd.json_normalize(data=cfg_dict, sep=".")

    cfg_small = {
        "benchmark_id": cfg.benchmark_id,
        "instance_set_id": cfg.instance_set_id,
        "test_set_id": Path(cfg.benchmark.config.test_set_path).name,
    }

    # Read performance data
    # Encoded in rewards
    # logs = load_logs(eval_dir / perf_fn)
    # perf_df = log2dataframe(logs, wide=True)

    # Read state data
    state_df = None
    if (eval_dir / state_fn).is_file():
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
    if state_df is not None:
        df = state_df.merge(reward_df)
    else:
        df = reward_df
    df = df.merge(action_df)

    for k, v in cfg_small.items():
        df[k] = v

    return df


def load_traineval_trajectories(path: str, train_instance_set_id: str) -> pd.DataFrame:
    path = Path(path)
    eval_dirs = list(path.glob(f"**/eval/{train_instance_set_id}/"))
    eval_dirs.sort()
    printr(eval_dirs)
    common_path = os.path.commonpath(eval_dirs)
    df_fn = Path("data") / common_path / "eval.csv"
    df_fn.parent.mkdir(parents=True, exist_ok=True)

    with Pool() as pool:
        dfs = pool.map(get_eval_df, eval_dirs)

    df = pd.concat(dfs).reset_index(drop=True)
    printr(df_fn)
    df.to_csv(df_fn, index=False)
    return df


if __name__ == "__main__":
    path, train_instance_set_id = Path("runs/Sigmoid/2D3M_train/ppo/full"), "sigmoid_2D3M_train"
    # path, train_instance_set_id = Path("runs/CMA-ES/seplow_train/ppo_sb3/full"), "train"
    df = load_traineval_trajectories(path=path, train_instance_set_id=train_instance_set_id)
