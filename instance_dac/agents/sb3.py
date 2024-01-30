from __future__ import annotations
import numpy as np
from pathlib import Path
import os

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


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        # return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")
        return os.path.join(self.save_path, f"{self.name_prefix}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

        return True


class SB3Agent(AbstractDACBenchAgent):
    def __init__(self, env, agent, logger: Logger):
        super().__init__(env)

        self.agent: BaseAlgorithm = agent(env=env)
        self.logger_callback = LoggerCallback(logger=logger)
        self.checkpoint_callback = CheckpointCallback(
            name_prefix="agent",
            save_freq=10000,
            save_path=Path(logger.output_path).parent,
        )

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
