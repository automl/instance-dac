# pylint: disable=function-redefined
from dacbench.abstract_agent import AbstractDACBenchAgent

import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

from jax.tree_util import tree_map, tree_structure

from pathlib import Path

# Policy definitions
def pi_func(env):
    def func_pi(S, is_training):
        shared = hk.Sequential((
            hk.Linear(8), 
            jax.nn.relu,
            hk.Linear(8), 
            jax.nn.relu,
        ))
        mu = hk.Sequential((
            shared,
            hk.Linear(8), 
            jax.nn.relu,
            hk.Linear(prod(4,), w_init=jnp.zeros),
            hk.Reshape((4,)),
        ))
        logvar = hk.Sequential((
            shared,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(9,), w_init=jnp.zeros),
            hk.Reshape((9,)),
        ))
        
        mu_s = mu(S)
        logvar_s = logvar(S)
        
        return ({'logits': mu_s}, {'logits': logvar_s} )

    return func_pi

# Value Function definition
def v_func(env):
    def func_v(S, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel
        ))
        return seq(S)
    return func_v

class PPO(AbstractDACBenchAgent):   
    """
        PPO agent for sigmoid
    """
    def __init__(self, env, seed):
        
        pi_f = pi_func(env)
        v_f  = v_func(env) 
        
        self.pi = coax.Policy(pi_f, env, random_seed=seed)
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