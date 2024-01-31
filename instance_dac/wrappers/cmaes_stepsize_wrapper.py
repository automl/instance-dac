from __future__ import annotations
from collections import OrderedDict

from gymnasium.core import ActionWrapper, Env
from gymnasium.spaces import Box
from typing import Any


class CMAESStepSizeWrapper(ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.action_space = Box(0.0, 10.0)

    def action(self, action: Any) -> Any:
        complete_action = OrderedDict({"step_size": action})
        return complete_action
