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
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper,  ObservationWrapper
from dacbench.abstract_env import AbstractEnv
from dacbench.abstract_agent import AbstractDACBenchAgent
from instance_dac.agent import PPO

import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax


def wrap_and_log(cfg: DictConfig, env: AbstractEnv) -> tuple[AbstractEnv, Logger]:
    experiment_name = "train" if not cfg.evaluate else "eval"
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
    for i in range(num_eval_episodes):
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


def train(env: AbstractEnv, agent: AbstractDACBenchAgent, num_episodes: int = 10):
    for i in range(num_episodes):
        done, truncated = False, False
        s, info = env.reset()

        while not (done or truncated):
            a, logp = agent.pi_targ(s, return_logp=True)
            s_next, r, done, truncated, info = env.step(a)

            # trace rewards
            agent.tracer.add(s, a, r, done or truncated, logp)
            while agent.tracer:
                agent.buffer.add(agent.tracer.pop())

            # learn
            if len(agent.buffer) >= agent.buffer.capacity:
                for _ in range(int(4 * agent.buffer.capacity / 32)):  # 4 passes per round
                    transition_batch = agent.buffer.sample(batch_size=32)
                    metrics_v, td_error = agent.simpletd.update(transition_batch, return_td_error=True)
                    metrics_pi = agent.ppo_clip.update(transition_batch, td_error)
                    env.record_metrics(metrics_v)
                    env.record_metrics(metrics_pi)

                agent.buffer.clear()
                agent.pi_targ.soft_update(pi, tau=0.1)

            if done or truncated:
                break

            s = s_next

    env.close()


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    benchmark = make_benchmark(cfg=cfg)
    env = benchmark.get_environment()

    env, logger = wrap_and_log(cfg, env)


    agent = PPO(env)

    if not cfg.evauate:
        train(env=env, agent=agent, num_episodes=cfg.num_episodes, logger=logger)
    else:
        evaluate(env, agent, cfg.num_eval_episodess)


if __name__ == "__main__":
    main()