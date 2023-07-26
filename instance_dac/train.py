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
from dacbench.abstract_agent import AbstractDACBenchAgent


def wrap_and_log(cfg: DictConfig, env: AbstractEnv) -> tuple[AbstractEnv, Logger]:
    ipath = Path(cfg.benchmark.config.instance_set_path)
    experiment_name = "train" if not cfg.evaluate else f"eval/{ipath.stem}"
    logger = Logger(
        experiment_name=experiment_name,
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


def evaluate(env: AbstractEnv, agent: AbstractDACBenchAgent, num_eval_episodes: int = 10):
    n_instances = len(env.instance_set)
    for i in range(num_eval_episodes * n_instances):
        env.reset()
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated):
            for a in [0, 1]:
                observation, reward, terminated, truncated, info = env.last()
                action = agent.act(state=observation, reward=reward)
                env.step(action)
            observation, reward, terminated, truncated, info = env.last()
            total_reward += reward


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    benchmark = make_benchmark(cfg=cfg)
    env = benchmark.get_environment()

    env, logger = wrap_and_log(cfg, env)

    if not cfg.evaluate:
        run_benchmark(env=env, agent=agent, num_episodes=cfg.num_episodes, logger=logger)
    else:
        # agent = load_agent(cfg)
        agent = RandomAgent(env=env)
        evaluate(env, agent, cfg.num_eval_episodes)


if __name__ == "__main__":
    main()