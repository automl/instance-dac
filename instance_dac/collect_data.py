from pathlib import Path
from rich import print as printr
from omegaconf import OmegaConf
import pandas as pd
from dacbench.logger import Logger, log2dataframe, load_logs
import matplotlib.pyplot as plt
from dacbench.plotting import plot_performance, plot_performance_per_instance, plot_state


cfg_fn = ".hydra/config.yaml"
perf_fn = "PerformanceTrackingWrapper.jsonl"
state_fn = "StateTrackingWrapper.jsonl"


if __name__ == "__main__":
    path = Path("runs/Sigmoid")
    eval_dirs = list(path.glob("**/eval/*"))
    eval_dirs.sort()
    printr(eval_dirs)

    for eval_dir in eval_dirs:
        # Get config
        cfg_filename = eval_dir / "../../.." / cfg_fn

        cfg = OmegaConf.load(cfg_filename)

        # Recover the correct test set path bc it gets overwritten
        cfg.benchmark.config.test_set_path = str(Path(cfg.benchmark.config.test_set_path).parent / (str(eval_dir.stem) + ".csv"))

        cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)

        cfg_dict_flat = pd.json_normalize(data=cfg_dict, sep=".")

        # Read performance data
        logs = load_logs(eval_dir / perf_fn)
        perf_df = log2dataframe(logs, wide=True)

        # Read state data
        logs = load_logs(eval_dir / state_fn)
        state_df = log2dataframe(logs, wide=True)
        printr(state_df)
        printr(cfg_dict_flat)


        break
