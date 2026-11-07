import jax
import jax.numpy as jnp
import pytest

from gfnx.environment.dag import DAGEnvironment
from gfnx.environment.hypergrid import HypergridEnvironment
from gfnx.environment.qm9_small import QM9SmallEnvironment
from gfnx.environment.tfbind import TFBind8Environment
from gfnx.metrics import ExactDistributionMetricsModule
from gfnx.reward.hypergrid import EasyHypergridRewardModule


def _uniform_policy(n_actions):
    def policy_fn(rng_key, obs, policy_params):
        return jnp.zeros((n_actions,)), {}

    return policy_fn


def _make_metric(env, reward_module=None):
    return ExactDistributionMetricsModule(
        metrics=["tv"],
        env=env,
        fwd_policy_fn=_uniform_policy(env.action_space.n),
        reward_module=reward_module,
        batch_size=8,
    )


@pytest.mark.parametrize("env_class", [TFBind8Environment, QM9SmallEnvironment])
def test_rejects_enumerable_env_without_state_helpers(env_class):
    """TFBind8/QM9 enumerate terminal states only, so construction must fail up front."""
    env = env_class()
    # They stay enumerable: ApproxDistributionMetricsModule relies on that.
    assert env.is_enumerable
    with pytest.raises(ValueError, match="get_all_states, state_to_index"):
        _make_metric(env)


def test_accepts_dag_environment():
    _make_metric(DAGEnvironment(num_variables=3))


def test_hypergrid_end_to_end():
    env = HypergridEnvironment(dim=2, side=4)
    env_params = env.init(jax.random.PRNGKey(0))
    reward_module = EasyHypergridRewardModule(side=4)
    reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())
    metric = _make_metric(env, reward_module)

    state = metric.init(
        jax.random.PRNGKey(0),
        metric.InitArgs(env_params=env_params, reward_params=reward_params),
    )
    state = metric.process(
        state,
        jax.random.PRNGKey(1),
        metric.ProcessArgs(policy_params=None, env_params=env_params),
    )
    assert jnp.isclose(state.exact_distribution.sum(), 1.0, atol=1e-5)
    assert jnp.isfinite(metric.get(state)["tv"])
