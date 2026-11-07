"""Tests for the Flow Consistency in Sub-graphs (FCS) metric."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfnx
from gfnx.base import BaseRewardModule
from gfnx.environment.hypergrid import HypergridEnvironment
from gfnx.metrics import (
    ExactDistributionMetricsModule,
    MultiMetricsModule,
    OnPolicyFCSMetricsModule,
    first_occurrence_mask,
    subset_total_variation,
)
from gfnx.reward.hypergrid import EasyHypergridRewardModule

DIM, SIDE = 2, 4


def uniform_policy_fn(rng_key, obs, policy_params):
    fwd_logits = jnp.zeros((DIM + 1,))
    bwd_logits = jnp.zeros((DIM,))
    return fwd_logits, {"forward_logits": fwd_logits, "backward_logits": bwd_logits}


def uniform_bwd_policy_fn(rng_key, obs, policy_params):
    fwd_logits = jnp.zeros((DIM + 1,))
    bwd_logits = jnp.zeros((DIM,))
    return bwd_logits, {"forward_logits": fwd_logits, "backward_logits": bwd_logits}


@chex.dataclass(frozen=True)
class TableRewardParams:
    log_rewards: chex.Array


class TableRewardModule(BaseRewardModule):
    """Reward given by a lookup table over hypergrid cells."""

    def __init__(self, env, env_params, log_rewards):
        self.env = env
        self.env_params = env_params
        self.log_rewards = log_rewards

    def init(self, rng_key, dummy_state):
        return TableRewardParams(log_rewards=self.log_rewards)

    def log_reward(self, state, reward_params):
        return reward_params.log_rewards[self.env.state_to_index(state, self.env_params)]

    def reward(self, state, reward_params):
        return jnp.exp(self.log_reward(state, reward_params))


@pytest.fixture(scope="module")
def setup():
    env = HypergridEnvironment(dim=DIM, side=SIDE)
    env_params = env.init(jax.random.PRNGKey(0))
    reward_module = EasyHypergridRewardModule(side=SIDE)
    reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

    # Exact marginal p_T of the uniform forward policy, by enumeration.
    exact_module = ExactDistributionMetricsModule(
        metrics=["tv"],
        env=env,
        fwd_policy_fn=uniform_policy_fn,
        reward_module=reward_module,
        batch_size=SIDE**DIM,
    )
    exact_state = exact_module.init(
        jax.random.PRNGKey(0),
        exact_module.InitArgs(env_params=env_params, reward_params=reward_params),
    )
    exact_state = exact_module.process(
        exact_state,
        jax.random.PRNGKey(0),
        exact_module.ProcessArgs(policy_params=None, env_params=env_params),
    )
    exact_tv = float(exact_module.get(exact_state)["tv"])

    # All terminal states, in `state_to_index` order.
    all_states = env.get_all_states(env_params)
    terminal_states = jax.vmap(lambda s: env.step(s, env.stop_action, env_params)[1])(all_states)
    log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(
        terminal_states, reward_params
    )
    return {
        "env": env,
        "env_params": env_params,
        "reward_module": reward_module,
        "reward_params": reward_params,
        "terminal_states": terminal_states,
        "exact_pt": np.asarray(exact_state.exact_distribution).reshape(-1),
        "true_dist": np.asarray(exact_state.true_distribution).reshape(-1),
        "log_rewards": np.asarray(log_rewards),
        "exact_tv": exact_tv,
    }


def _reference_subset_tv(p, r, idx, deduplicate=True):
    if deduplicate:
        idx = np.unique(idx)
    p, r = p[idx], r[idx]
    return 0.5 * np.abs(p / p.sum() - r / r.sum()).sum()


def test_subset_total_variation_matches_manual():
    log_p = jnp.log(jnp.array([0.5, 0.3, 0.2, 0.7]))
    log_r = jnp.log(jnp.array([1.0, 1.0, 2.0, 100.0]))
    mask = jnp.array([True, True, True, False])
    # Restricted to the first three elements: p = [.5, .3, .2], r = [.25, .25, .5].
    expected = 0.5 * (0.25 + 0.05 + 0.3)
    np.testing.assert_allclose(subset_total_variation(log_p, log_r, mask), expected, rtol=1e-6)


def test_subset_total_variation_zero_when_proportional():
    log_p = jax.random.normal(jax.random.PRNGKey(0), (16,))
    mask = jnp.arange(16) % 3 != 0
    tv = subset_total_variation(log_p, log_p + 7.3, mask)
    np.testing.assert_allclose(tv, 0.0, atol=1e-6)


def test_first_occurrence_mask(setup):
    idx = jnp.array([3, 5, 3, 0, 5, 5, 7])
    states = jax.tree.map(lambda x: x[idx], setup["terminal_states"])
    mask = first_occurrence_mask(states)
    np.testing.assert_array_equal(mask, [True, True, False, True, False, False, True])


@pytest.mark.parametrize("deduplicate", [True, False])
def test_tv_per_subset_matches_exact(setup, deduplicate):
    module = OnPolicyFCSMetricsModule(
        env=setup["env"],
        reward_module=setup["reward_module"],
        fwd_policy_fn=uniform_policy_fn,
        bwd_policy_fn=uniform_bwd_policy_fn,
        subset_size=6,
        n_subsets=4,
        n_rounds=2000,
        batch_size=24,
        deduplicate=deduplicate,
    )
    # Fixed subsets (with duplicates in the first two).
    idx = np.array([
        [0, 5, 5, 10, 15, 3],
        [1, 1, 1, 2, 7, 12],
        [4, 6, 8, 9, 11, 13],
        [0, 15, 3, 12, 5, 10],
    ])
    subset_states = jax.tree.map(lambda x: x[idx], setup["terminal_states"])
    tv = jax.jit(module.compute_tv_per_subset)(
        jax.random.PRNGKey(1),
        subset_states,
        None,
        setup["env_params"],
        setup["reward_params"],
    )
    rewards = np.exp(setup["log_rewards"])
    if deduplicate:
        expected = [_reference_subset_tv(setup["exact_pt"], rewards, row) for row in idx]
    else:
        # Multiset restriction: each copy carries its own mass.
        expected = [
            0.5
            * np.abs(
                setup["exact_pt"][row] / setup["exact_pt"][row].sum()
                - rewards[row] / rewards[row].sum()
            ).sum()
            for row in idx
        ]
    np.testing.assert_allclose(tv, expected, atol=0.02)


def test_full_test_set_recovers_tv(setup):
    """With beta = |X|, FCS equals TV(p_T, R/Z) (Theorem 5 / Corollary 1)."""
    n_states = SIDE**DIM
    module = gfnx.metrics.TestFCSMetricsModule(
        env=setup["env"],
        reward_module=setup["reward_module"],
        bwd_policy_fn=uniform_bwd_policy_fn,
        subset_size=n_states,
        n_subsets=2,
        n_rounds=2000,
    )
    state = module.init(jax.random.PRNGKey(0), module.InitArgs(test_set=setup["terminal_states"]))
    process_args = module.ProcessArgs(
        policy_params=None,
        env_params=setup["env_params"],
        reward_params=setup["reward_params"],
    )
    state = jax.jit(module.process)(state, jax.random.PRNGKey(2), process_args)
    fcs = float(module.get(state)["fcs"])
    np.testing.assert_allclose(fcs, setup["exact_tv"], atol=0.02)
    assert setup["exact_tv"] > 0.1  # the uniform policy is far from the target


def test_test_set_smaller_than_subset_raises(setup):
    module = gfnx.metrics.TestFCSMetricsModule(
        env=setup["env"],
        reward_module=setup["reward_module"],
        bwd_policy_fn=uniform_bwd_policy_fn,
        subset_size=SIDE**DIM + 1,
    )
    with pytest.raises(ValueError, match="fewer than"):
        module.init(jax.random.PRNGKey(0), module.InitArgs(test_set=setup["terminal_states"]))


def test_on_policy_fcs_is_zero_for_matching_reward(setup):
    """If R is proportional to p_T, FCS is zero (up to Monte Carlo error)."""
    env, env_params = setup["env"], setup["env_params"]
    matching_reward = TableRewardModule(env, env_params, jnp.log(setup["exact_pt"]) + 3.0)
    matching_params = matching_reward.init(jax.random.PRNGKey(0), env.reset())

    def make_module(reward_module):
        return OnPolicyFCSMetricsModule(
            env=env,
            reward_module=reward_module,
            fwd_policy_fn=uniform_policy_fn,
            bwd_policy_fn=uniform_bwd_policy_fn,
            subset_size=8,
            n_subsets=8,
            n_rounds=500,
        )

    scores = {}
    for name, reward_module, reward_params in [
        ("matching", matching_reward, matching_params),
        ("easy", setup["reward_module"], setup["reward_params"]),
    ]:
        module = make_module(reward_module)
        state = module.init(jax.random.PRNGKey(0))
        state = jax.jit(module.process)(
            state,
            jax.random.PRNGKey(3),
            module.ProcessArgs(
                policy_params=None, env_params=env_params, reward_params=reward_params
            ),
        )
        scores[name] = float(module.get(state)["fcs"])
    assert scores["matching"] < 0.03
    assert scores["easy"] > 0.1


def test_step_inside_multimetrics_under_jit(setup):
    module = MultiMetricsModule({
        "fcs": OnPolicyFCSMetricsModule(
            env=setup["env"],
            reward_module=setup["reward_module"],
            fwd_policy_fn=uniform_policy_fn,
            bwd_policy_fn=uniform_bwd_policy_fn,
            subset_size=4,
            n_subsets=3,
            n_rounds=2,
        )
    })
    state = module.init(jax.random.PRNGKey(0), module.InitArgs(metrics_args={}))
    update_args = module.UpdateArgs(metrics_args={})
    process_args = module.ProcessArgs(
        metrics_args={
            "fcs": OnPolicyFCSMetricsModule.ProcessArgs(
                policy_params=None,
                env_params=setup["env_params"],
                reward_params=setup["reward_params"],
            )
        }
    )

    @jax.jit
    def step(idx, state, prev_eval_info):
        return module.step(
            idx=idx,
            metrics_state=state,
            rng_key=jax.random.PRNGKey(4),
            update_args=update_args,
            process_args=process_args,
            eval_each=2,
            num_train_steps=10,
            prev_eval_info=prev_eval_info,
        )

    prev = {"fcs/fcs": jnp.array(-1.0)}
    _, info = step(1, state, prev)  # not an eval step
    assert float(info["fcs/fcs"]) == -1.0
    _, info = step(2, state, prev)  # eval step
    assert 0.0 <= float(info["fcs/fcs"]) <= 1.0
