from __future__ import annotations

import tqdm
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
from dacbench.wrappers import PerformanceTrackingWrapper, StateTrackingWrapper,  ObservationWrapper, ActionFrequencyWrapper
from dacbench.abstract_env import AbstractEnv
from dacbench.abstract_agent import AbstractDACBenchAgent
from instance_dac.agent import PPO
from instance_dac.wrapper import RewardTrackingWrapper

import coax


def wrap_and_log(cfg: DictConfig, env: AbstractEnv) -> tuple[AbstractEnv, Logger]:
    ipath = Path(cfg.benchmark.config.test_set_path)
    experiment_name = "train" if not cfg.evaluate else f"eval/{ipath.stem}"
    logger = Logger(
        experiment_name=experiment_name,
        output_path=Path("logs"),
        step_write_frequency=None,
        episode_write_frequency=None,
    )
    state_logger = logger.add_module(StateTrackingWrapper)
    performance_logger = logger.add_module(PerformanceTrackingWrapper)
    action_logger = logger.add_module(ActionFrequencyWrapper)
    reward_logger = logger.add_module(RewardTrackingWrapper)

    env = PerformanceTrackingWrapper(env, logger=performance_logger)
    env = StateTrackingWrapper(env, logger=state_logger)
    env = ActionFrequencyWrapper(env, logger=action_logger)
    env = RewardTrackingWrapper(env, logger=reward_logger)
    env = coax.wrappers.TrainMonitor(env, name=experiment_name)

    # Add env to logger
    logger.set_env(env)

    assert logger.env is not None

    return env, logger


def evaluate(env: AbstractEnv, agent: AbstractDACBenchAgent, logger: Logger = None, num_eval_episodes: int = 10):
    if logger is not None:
        logger.reset_episode()
        logger.set_env(env)
    
    n_instances = len(env.instance_set)
    for i in tqdm.tqdm(range(num_eval_episodes * n_instances)):
        observation, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0
        reward = None
        while not (terminated or truncated):
            # observation, reward, terminated, truncated, info = env.last()
            action = agent.act(state=observation, reward=reward)
            observation, reward, terminated, truncated, info = env.step(action)
            # observation, reward, terminated, truncated, info = env.last()
            total_reward += reward
            if logger is not None:
                logger.next_step()
        if logger is not None:
            logger.next_episode()
    env.close()


def train(env: AbstractEnv, agent: AbstractDACBenchAgent, logger: Logger = None, num_episodes: int = 10):
    if logger is not None:
        logger.reset_episode()
        logger.set_env(env)

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
                agent.pi_targ.soft_update(agent.pi, tau=0.1)

            if logger is not None:
                logger.next_step()

            if done or truncated:
                break

            s = s_next
            
        if logger is not None:
            logger.next_episode()

    if logger is not None:
        agent.save(Path(logger.output_path).parent)

    env.close()


@hydra.main(config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    benchmark = make_benchmark(cfg=cfg)
    env = benchmark.get_environment()

    env, logger = wrap_and_log(cfg, env)

    # import pdb 
    # print(env.action_space)
    # pdb.set_trace()

    agent = PPO(env, seed=cfg.seed)

    if not cfg.evaluate:
        train(env=env, agent=agent, num_episodes=cfg.num_episodes, logger=logger)
    else:
        env.use_test_set()
        agent.load(Path(logger.output_path).parent)
        evaluate(env, agent, logger, cfg.num_eval_episodes)


if __name__ == "__main__":
    main()