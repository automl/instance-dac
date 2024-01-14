from __future__ import annotations
import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.logger import Logger


class LoggerCallback(BaseCallback):
    def __init__(self, logger: Logger | None, verbose: int = 0):
        super().__init__(verbose)

        self.db_logger = logger

    def _on_step(self) -> bool:
        if self.db_logger is not None:
            self.db_logger.next_step()

            # episode end
            if np.any(self.locals["dones"]):  
                self.db_logger.next_episode()

        return super().on_step()
    


class SB3Agent(AbstractDACBenchAgent):
    def __init__(self, env, agent, logger: Logger):
        super().__init__(env)

        self.agent = agent(env=env)
        self.logger_callback = LoggerCallback(logger=logger)

    def act(self, state, reward):
        action, state = self.agent.predict(observation=state) 
        return action
    
    def train(self, next_state, reward):
        return super().train(next_state, reward)
    
    def end_episode(self, state, reward):
        return super().end_episode(state, reward)
    
    def save(self, path: Path):
        save_path = path / f"agent.zip"
        self.agent.save(save_path)

    def load(self, path: Path):
        self.agent = self.agent.load(path / f"agent.zip")
    