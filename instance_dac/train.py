from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from instance_dac.make import make_benchmark

from pathlib import Path

from dacbench.plotting import plot_performance, plot_performance_per_instance
from dacbench.logger import Logger, log2dataframe, load_logs
from dacbench.agents.simple_agents import RandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.runner import run_benchmark
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper
from dacbench.abstract_env import AbstractEnv



# Run an experiment and log the results
if __name__ == "__main__":

    # Make benchmark
    bench = SigmoidBenchmark()

    # Run for 10 episodes each on 10 seeds
    num_episodes = 10
    seeds = range(10)

    # Make logger object and add modules for performance & state logging
    logger = Logger(
        experiment_name="sigmoid_example",
        output_path=Path("plotting/data"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    state_logger = logger.add_module(StateTrackingWrapper)
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    for s in seeds:

        # Make & wrap benchmark environment
        env = bench.get_benchmark(seed=s)
        env = PerformanceTrackingWrapper(env, logger=performance_logger)
        env = StateTrackingWrapper(env, logger=state_logger)

        # Add env to logger
        logger.set_env(env)

        # Run random agent
        agent = RandomAgent(env)
        run_benchmark(env, agent, num_episodes, logger)

    # Close logger object
    logger.close()

def wrap_and_log(cfg: DictConfig, env: AbstractEnv) -> tuple[AbstractEnv, Logger]:
    logger = Logger(
        experiment_name="train",
        output_path=Path("logs"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    state_logger = logger.add_module(StateTrackingWrapper)
    performance_logger = logger.add_module(PerformanceTrackingWrapper)

    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    env = StateTrackingWrapper(env, logger=state_logger)

    # Add env to logger
    logger.set_env(env)

    return env, logger


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    benchmark = make_benchmark(cfg=cfg)
    env = benchmark.get_environment()

    env, logger = wrap_and_log(cfg, env)

    run_benchmark(env=env, agent=agent, num_episodes=cfg.num_episodes, logger=logger)


if __name__ == "__main__":
    main()