from __future__ import annotations

import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rich import inspect
from instance_dac.make import make_benchmark
import gymnasium
from gymnasium.wrappers import FlattenObservation, NormalizeObservation

from pathlib import Path

from dacbench.plotting import plot_performance, plot_performance_per_instance
from dacbench.logger import Logger, log2dataframe, load_logs
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper,  ObservationWrapper, ActionFrequencyWrapper
from dacbench.abstract_env import AbstractEnv
from dacbench.abstract_agent import AbstractDACBenchAgent
from instance_dac.agent import PPO
from instance_dac.wrapper import RewardTrackingWrapper

import coax


# @hydra.main(config_path="configs", config_name="base.yaml")
# def main(cfg: DictConfig) -> None:
#     cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
#     printr(cfg_dict)
    
import argparse
import os
from subprocess import Popen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args, unknown_args = parser.parse_known_args()  # unknown args are hydra commands
    # unknown_args = [f"'{a}'" for a in unknown_args]

    add_multirun_flag = False
    if unknown_args[-1] == "-m":
        unknown_args.pop(-1)
        add_multirun_flag = True

    if add_multirun_flag:
        unknown_args += ["-m"]

    cmd = [
        "python",
        "instance_dac/train.py",
    ] + unknown_args

    print(cmd)
    print(" ".join(cmd))

    if not args.dry:
        env = os.environ.copy()
        p = Popen(cmd, env=env)
        p.communicate()

if __name__ == "__main__":
    main()