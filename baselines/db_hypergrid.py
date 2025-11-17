"""Single-file implementation for Detailed Balane algorithm in hypergrid environment

TODO: Define all __init__.py to improve readability
"""

import chex
import functools
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any, NamedTuple

from omegaconf import OmegaConf

import gfnx
import gfnx.environment
import gfnx.environment.base
import gfnx.environment.environment
import gfnx.environment.environment.hypergrid
import gfnx.environment.reward
import gfnx.environment.reward.hypergrid
import gfnx.environment.sampling
import gfnx.environment.utils
import gfnx.environment.utils.masking



class MLPPolicy(eqx.Module):
    """
    3-layer NN used for estimation of policies and flow
    """
    network: eqx.nn.MLP
    train_backward_policy: bool

    def __init__(
        self,
        dim: int,
        hidden_size: int,
        train_backward_policy: bool,
        rng_key: chex.PRNGKey
    ):
        self.train_backward_policy = train_backward_policy

        output_size = dim + 2
        if train_backward_policy:
            output_size += dim
        self.network = eqx.nn.MLP(
            in_size = dim+1, 
            out_size=output_size,
            width_size=hidden_size,
            depth=3,
            key=rng_key
        )
    
    def __call__(self, x : chex.Array) -> chex.Array:
        x = self.network(x)
        if self.train_backward:
            forward_logits, flow, backward_logits = jnp.split(
                x, [self.dim + 1, self.dim + 2], axis=-1
            )
        else:
            forward_logits, flow = jnp.split(x, [self.dim + 1], axis=-1)
            backward_logits = jnp.zeros(shape=(x.shape[0], self.dim), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "flow": flow.squeeze(-1),
            "backward_logits": backward_logits
        }

# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.environment.environment.hypergrid.HypergridEnvironment
    env_params: chex.Array
    model: MLPPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics: chex.Array

@chex.dataclass
class TrajectoryData:
    obs: gfnx.environment.base.TObs
    state: gfnx.environment.base.TEnvState
    action: gfnx.environment.base.TAction
    log_gfn_reward: chex.Array
    done: chex.Array
    pad: chex.Array
    info: dict

@chex.dataclass
class TransitionData:
    obs: gfnx.environment.base.TObs  # [B x ...]
    state: gfnx.environment.base.TEnvState  # [B x ...]
    action: gfnx.environment.base.TAction  # [B]
    log_gfn_reward: chex.Array  # [B]
    next_obs: gfnx.environment.base.TObs  # [B x ...]
    next_state: gfnx.environment.base.TEnvState  # [B x ...]
    done: chex.Array  # [B]
    pad: chex.Array  # [B]



def sample_forward_trajectories(
    rng_key: chex.PRNGKey,
    num_envs: int,
    policy: MLPPolicy,
    env: gfnx.environment.base.TEnvironment,
    env_params: gfnx.environment.base.TEnvParams
):
    env_obs, env_state = env.reset(num_envs, env_params=env_params)
    # Partition the policy to pass into jax.lax.scan
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    @chex.dataclass
    class TrajSamplingState:
        env_obs: gfnx.environment.base.TObs
        env_state: gfnx.environment.base.TEnvState

        rng_key: chex.PRNGKey
        policy_params: Any
        env_params: gfnx.environment.base.TEnvParams

    @functools.partial(jax.jit, donate_argnums=(1,))
    def environment_step_fn(
        traj_step_state: TrajSamplingState, _: None
    ) -> tuple[TrajSamplingState, TrajectoryData]:
        # Unpack the sampling state
        policy = eqx.combine(policy_params, policy_static)
        env_params = traj_step_state.env_params
        env_state = traj_step_state.env_state
        pad = env_state.is_done

        env_obs = traj_step_state.env_obs
        rng_key = traj_step_state.rng_key

        rng_key, sample_rng_key = jax.random.split(rng_key)
        # Call the network, note: it is not batched by default
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
        fwd_logits = policy_outputs["forward_logits"]
        # Very important part: masking invalid actions
        invalid_mask = env.get_invalid_mask(env_state, env_params)
        masked_fwd_logits = gfnx.environment.utils.masking.mask_logits(
            fwd_logits, invalid_mask
        )
        # Mask the second time to ensure 0.0 probability for invalid actions
        policy_probs = jax.nn.softmax(masked_fwd_logits, axis=-1)

        info = {
            "entropy": -jnp.sum(
                policy_probs * jax.nn.log_softmax(masked_fwd_logits), axis=-1
            )
        }
        action = env.sample_action(sample_rng_key, policy_probs)
        next_obs, next_env_state, log_gfn_reward, done, _ = env.step(
            env_state, action, env_params
        )
        traj_data = TrajectoryData(
            obs=env_obs,
            state=env_state,
            action=action,
            log_gfn_reward=log_gfn_reward,
            done=done,
            pad=pad,
            info=info
        )
        next_traj_state = traj_step_state.replace(
            env_obs=next_obs,
            env_state=next_env_state,
            rng_key=rng_key,
        )

        return next_traj_state, traj_data

    final_traj_stats, traj_data = jax.lax.scan(
        f=environment_step_fn,
        init=TrajSamplingState(
            env_obs=env_obs,
            env_state=env_state,
            rng_key=rng_key,
            policy_params=policy_params,
            env_params=env_params
        ),
        xs=None,
        length=env.max_steps_in_episode + 1
    )

    # Now, the shape of traj data is [(T + 1) x B x ...]
    # Need to transpose it to [B x (T + 1) x ...]
    chex.assert_tree_shape_prefix(traj_data, (env.max_steps_in_episode + 1, num_envs))
    traj_data = jax.tree_map(
        lambda x: jnp.transpose(x, axes=(1,0) + tuple(range(2, x.ndim))), 
        traj_data
    )
    chex.assert_tree_shape_prefix(traj_data, (num_envs, env.max_steps_in_episode + 1))

    # Logging data
    final_env_state = final_traj_stats.env_state
    traj_entropy = jnp.sum(
        jnp.where(traj_data.pad, 0.0, traj_data.info["entropy"]), axis=1
    )
    return traj_data, {'entropy': traj_entropy, 'final_env_state': final_env_state}

def split_traj_to_transitions(traj_data: TrajectoryData) -> TransitionData:
    # TODO: provide some general function, that will split the trajectory to transitions
    # by names
    def slice_prev(tree):
        return jax.tree_map(lambda x: x[:, :-1], tree)

    def slice_next(tree):
        return jax.tree_map(lambda x: x[:, 1:], tree)

    base_transition_data = TransitionData(
        obs=slice_prev(traj_data.obs),
        state=slice_prev(traj_data.state),
        action=slice_prev(traj_data.action),
        log_gfn_reward=slice_prev(traj_data.log_gfn_reward),
        next_obs=slice_next(traj_data.obs),
        next_state=slice_next(traj_data.state),
        done=slice_prev(traj_data.done),
        pad=slice_prev(traj_data.pad)
    )
    # Reshape all the arrays to [BT x ...]
    return jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), base_transition_data)

@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    traj_data, log_info = sample_forward_trajectories(
        sample_traj_key,
        num_envs,
        train_state.model,
        train_state.env,
        train_state.env_params
    )
    transitions = split_traj_to_transitions(traj_data)
    bwd_actions = train_state.env.get_backward_action(
        transitions.state,
        transitions.action,
        transitions.next_state,
        train_state.env_params
    )

    # Step 2. Compute the loss
    def loss_fn(model : MLPPolicy) -> chex.Array:
        # Call the network to get the logits
        policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.obs)
        # Compute the forward log-probs
        fwd_logits = policy_outputs["forward_logits"]
        invalid_mask = env.get_invalid_mask(transitions.state, env_params)
        masked_fwd_logits = gfnx.environment.utils.masking.mask_logits(fwd_logits, invalid_mask)
        fwd_all_log_probs = jax.nn.log_softmax(masked_fwd_logits, axis=-1)
        fwd_logprobs = jnp.take_along_axis(
            fwd_all_log_probs, 
            jnp.expand_dims(transitions.action, axis=-1), 
            axis=-1
        ).squeeze(-1)
        fwd_flow = policy_outputs["flow"]

        # Compute the stats for the next state
        next_policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.next_obs)
        bwd_logits = next_policy_outputs["backward_logits"]
        next_bwd_invalid_mask = env.get_invalid_backward_mask(
            transitions.next_state,
            env_params
        )
        masked_bwd_logits = gfnx.environment.utils.masking.mask_logits(bwd_logits, next_bwd_invalid_mask)
        bwd_all_log_probs = jax.nn.log_softmax(masked_bwd_logits, axis=-1)
        bwd_logprobs = jnp.take_along_axis(
            bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
        ).squeeze(-1)
        next_flow = next_policy_outputs["flow"]
        # Replace the target with the log_gfn_reward if the episode is done
        target = jnp.where(
            transitions.done,
            transitions.log_gfn_reward, 
            bwd_logprobs + next_flow
        )

        # Compute the DB loss with masking
        loss = optax.l2_loss(
            jnp.where(transitions.pad, 0.0, fwd_logprobs + fwd_flow),
            jnp.where(transitions.pad, 0.0, target)
        ).mean()
        return loss, log_info

    (mean_loss, log_info), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(train_state.model)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, eqx.filter(train_state.model, eqx.is_array)
    )
    model = eqx.apply_updates(train_state.model, updates)

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=model,
        opt_state=opt_state
    )

@hydra.main(config_path="configs/", config_name="db_hypergrid")
def run_experiment(cfg: OmegaConf) -> None:
    rng_key = jax.random.key(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Define the reward function for the environment
    if cfg.environment.reward == "easy":
        reward_module = gfnx.environment.reward.hypergrid.EasyRewardModule()
    elif cfg.environment.reward == "hard":
        reward_module = gfnx.environment.reward.hypergrid.HardRewardModule()
    else:
        raise ValueError(f"Unknown reward function {cfg.environment.reward}")

    # Initialize the environment and its inner parameters
    env = gfnx.environment.environment.hypergrid.HypergridEnvironment(
        reward_module,
        dim=cfg.environment.dim,
        side=cfg.environment.side
    )
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    # TODO: change it for the dimension of observation space
    model = MLPPolicy(
        dim=cfg.environment.dim,
        hidden_size=cfg.network.hidden_size,
        train_backward_policy=False,
        rng_key=net_init_key
    )

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # TODO: initialize problem-dependent metrics here
    metrics = None
    del eval_init_key

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics=metrics
    )
    # Split initial train state into parameters and static parts to make jit work
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    def train_step_wrapper(idx: int, train_state_params):
        # Wrapper to avoid a usual unfiltered jit in JAX
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    # Run the training loop via JAX lax.fori_loop
    train_state_params = jax.lax.fori_loop(
        lower=0, upper=cfg.num_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params
    )


if __name__ == "__main__":
    run_experiment()

