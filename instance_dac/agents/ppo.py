# pylint: disable=function-redefined
from dacbench.abstract_agent import AbstractDACBenchAgent

import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from jax.tree_util import tree_map, tree_structure

from pathlib import Path

# Policy definitions


def make_func_pi(env):
    def func_pi(S, is_training):
        shared = hk.Sequential(
            (
                hk.Linear(8),
                jax.nn.relu,
                hk.Linear(8),
                jax.nn.relu,
            )
        )

        if isinstance(env.action_space, Discrete):
            logits = hk.Sequential(
                (
                    shared,
                    hk.Linear(8),
                    jax.nn.relu,
                    hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
                    hk.Reshape(env.action_space.shape),
                )
            )
            return {"logits": logits}
        elif isinstance(env.action_space, MultiDiscrete):
            output = []
            for n in env.action_space.nvec:
                logits = hk.Sequential(
                    (
                        shared,
                        hk.Linear(8),
                        jax.nn.relu,
                        hk.Linear(prod(n), w_init=jnp.zeros),
                        hk.Reshape((n,)),
                    )
                )
                output.append({"logits": logits(S)})
            return tuple(output)
        elif isinstance(env.action_space, Box):
            mu = hk.Sequential(
                (
                    shared,
                    hk.Linear(8),
                    jax.nn.relu,
                    hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
                    hk.Reshape(env.action_space.shape),
                )
            )
            logvar = hk.Sequential(
                (
                    shared,
                    hk.Linear(8),
                    jax.nn.relu,
                    hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
                    hk.Reshape(env.action_space.shape),
                )
            )
            # TODO automatic design for cont/discrete actions
            return {"mu": mu(S), "logvar": logvar(S)}
        else:
            raise ValueError(f"Action space {type(env.action_space)} is not supported.")

    return func_pi


# Value Function definition
def v_func(env):
    def func_v(S, is_training):
        seq = hk.Sequential(
            (
                hk.Linear(8),
                jax.nn.relu,
                hk.Linear(8),
                jax.nn.relu,
                hk.Linear(8),
                jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros),
                jnp.ravel,
            )
        )
        return seq(S)

    return func_v


class PPO(AbstractDACBenchAgent):
    """
    PPO agent for sigmoid
    """

    def __init__(self, env, seed):
        pi_f = make_func_pi(env)
        v_f = v_func(env)

        self.pi = coax.Policy(pi_f, env, random_seed=seed, proba_dist=coax.proba_dists.ProbaDist(env.action_space))
        self.v = coax.V(v_f, env, random_seed=seed)

        # target network
        self.pi_targ = self.pi.copy()

        # experience tracer
        self.tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=512)

        # policy regularizer (avoid premature exploitation)
        self.policy_reg = coax.regularizers.EntropyRegularizer(self.pi, beta=0.01)

        # updaters
        self.simpletd = coax.td_learning.SimpleTD(self.v, optimizer=optax.adam(1e-3))
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, regularizer=self.policy_reg, optimizer=optax.adam(1e-4))

    def act(self, state, reward):
        return self.pi_targ(state, return_logp=False)

    def save(self, path: Path):
        save_path = path / f"agent.pkl.lz4"
        coax.utils.dump(self.__dict__, save_path)

    def load(self, path: Path):
        self.__dict__ = coax.utils.load(path / f"agent.pkl.lz4")
