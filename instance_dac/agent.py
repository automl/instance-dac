# pylint: disable=function-redefined
from dacbench.abstract_agent import AbstractDACBenchAgent

import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

# Policy definitions
def func_pi(S, is_training):
    shared = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
    ))
    mu = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    logvar = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    return {'mu': mu(S), 'logvar': logvar(S)}


# Value Function definition
def func_v(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return seq(S)


class PPO(AbstractDACBenchAgent):   
    """
        PPO agent for sigmoid
    """
    def __init__(self, env):
        
        
        self.pi = coax.Policy(func_pi, env)
        self.v = coax.V(func_v, env)

        # target network
        self.pi_targ = self.pi.copy()

        # experience tracer
        self.tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=512)


        # policy regularizer (avoid premature exploitation)
        self.policy_reg = coax.regularizers.EntropyRegularizer(pi, beta=0.01)


        # updaters
        self.simpletd = coax.td_learning.SimpleTD(v, optimizer=optax.adam(1e-3))
        self.ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg, optimizer=optax.adam(1e-4))
        

    def act(self, state, reward):
        
        return self.pi_targ(state, return_false=True)
