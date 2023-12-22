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

from instance_dac.utils.oracle_set_generation import generate_oracle_set

import coax
from hydra import compose, initialize
from omegaconf import OmegaConf


# @hydra.main(config_path="configs", config_name="base.yaml")
# def main(cfg: DictConfig) -> None:
#     cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
#     printr(cfg_dict)
    
import argparse
import os
from subprocess import Popen
import subprocess
from rich import print as printr
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args, unknown_args = parser.parse_known_args()  # unknown args are hydra commands
    # unknown_args = [f"'{a}'" for a in unknown_args]

    # log = logging.getLogger("Dispatch")
    
    add_multirun_flag = False
    if unknown_args:
        if unknown_args[-1] == "-m":
            unknown_args.pop(-1)
            add_multirun_flag = True

        unknown_args = [
            f"'{a}'" if "range" in a else a for a in unknown_args  
        ]
    printr("-"*50)
    printr("Hydra Overrides")
    printr(unknown_args)


    if args.oracle:
        with initialize(version_base=None, config_path="configs"):
            _overrides = [o for o in unknown_args if ("benchmark" in o or "inst" in o)]
            cfg = compose(config_name="base", overrides=_overrides)
            printr("-"*50)
            printr("Source Config")
            printr(cfg)
            printr("-"*50)
        
        override, n_instances = generate_oracle_set(
            instance_set_path=cfg.benchmark.config.instance_set_path,
            instance_set_id=cfg.instance_set_id,
            benchmark_id=cfg.benchmark_id,
        )
        # Remove inst set and add new override
        unknown_args = [o for o in unknown_args if "inst" not in o]
        unknown_args.append(override)
        unknown_args.append("'instance_set_selection=oracle'")

        printr(f"Found {n_instances} instances.")
        printr("-"*50)

    if add_multirun_flag:
        unknown_args += ["-m"]

    cmd = [
        "python",
        "instance_dac/train.py",
    ] + unknown_args

    printr("Command")
    printr(" ".join(cmd))
    printr("-"*50)

    if not args.dry:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        p = Popen(" ".join(cmd), env=env, shell=True)
        p.communicate()

if __name__ == "__main__":
    main()