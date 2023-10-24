from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from gymnasium import Wrapper, spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class RewardTrackingWrapper(Wrapper):
    """
    Wrapper to track the reward.
    """

    def __init__(self, env, logger=None):
        """
        Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        logger: logger.ModuleLogger
            logger to write to

        """
        super(RewardTrackingWrapper, self).__init__(env)
        self.overall_rewards = []
        self.logger = logger

    def __setattr__(self, name, value):
        """
        Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in [
            "overall_rewards",
            "env",
            "step",
            "logger",
            "get_rewards",
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        """
        Get attribute value of wrapper if available and of env if not.

        Parameters
        ----------
        name : str
            Attribute to get

        Returns
        -------
        value
            Value of given name

        """
        if name in [
            "overall_rewards",
            "env",
            "step",
            "logger",
            "get_rewards",
        ]:
            return object.__getattribute__(self, name)

        else:
            return getattr(self.env, name)

    def step(self, action):
        """
        Execute environment step and record state.

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, metainfo

        """
        state, reward, terminated, truncated, info = self.env.step(action)
        self.overall_rewards.append(reward)
        if self.logger is not None:
            self.logger.log_space("reward", reward)

        return state, reward, terminated, truncated, info

    def get_rewards(self):
        """
        Get state progression.

        Returns
        -------
        np.array or np.array, np.array
            all states or all states and interval sorted states

        """
        return self.overall_rewards
