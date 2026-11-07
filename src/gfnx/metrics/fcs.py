r"""Flow Consistency in Sub-graphs (FCS) metric.

FCS was introduced by Silva et al., "When do GFlowNets learn the right distribution?"
(ICLR 2025) as a tractable, provably sound goodness-of-fit measure for GFlowNets.
Instead of comparing the full (intractable) distributions, it compares the
learned marginal $p_T$ and the reward $R$ *restricted* to random subsets
$S \subseteq \mathcal{X}$ of terminal states:

$$
p_T^{(S)}(x) = \frac{\mathbb{1}\{x \in S\}\, p_T(x)}{\sum_{y \in S} p_T(y)}, \qquad
R^{(S)}(x) = \frac{\mathbb{1}\{x \in S\}\, R(x)}{\sum_{y \in S} R(y)},
$$

$$
\mathrm{FCS}(p_T, R) = \mathbb{E}_{S \sim P_S}\left[\mathrm{TV}\big(p_T^{(S)}, R^{(S)}\big)\right].
$$

FCS lies in $[0, 1]$ and equals zero iff $p_T \propto R$ (for any full-support $P_S$).
The subset size $\beta$ interpolates between a ratio-matching metric ($\beta = 2$) and
the TV distance ($\beta = |\mathcal{X}|$). The marginal $p_T(x)$ is estimated by
importance sampling with the backward policy
(see [`estimate_marginal_log_probs`][gfnx.utils.marginal.estimate_marginal_log_probs]).
"""

from abc import abstractmethod
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..base import TEnvironment, TEnvParams, TEnvState, TRewardModule, TRewardParams
from ..utils.marginal import estimate_marginal_log_probs
from ..utils.masking import masked_sum
from ..utils.rollout import TPolicyFn, TPolicyParams, forward_rollout
from .base import (
    BaseInitArgs,
    BaseMetricsModule,
    BaseProcessArgs,
    EmptyInitArgs,
    EmptyUpdateArgs,
    MetricsState,
)


def subset_total_variation(
    log_probs: Float[Array, " subset_size"],
    log_rewards: Float[Array, " subset_size"],
    mask: Bool[Array, " subset_size"],
) -> Float[Array, ""]:
    r"""TV distance between $p_T$ and $R$ restricted to one subset $S$.

    Both inputs are unnormalised log-masses; each is renormalised over the elements
    of the subset before taking the total variation, so any constant offset (e.g. an
    unknown $\log Z$) cancels.

    Args:
        log_probs: Log-probabilities $\log p_T(x)$ of the subset elements.
        log_rewards: Log-rewards $\log R(x)$ of the subset elements.
        mask: `True` for the elements that belong to $S$ (e.g. excludes duplicates).

    Returns:
        $\mathrm{TV}(p_T^{(S)}, R^{(S)}) \in [0, 1]$.
    """
    chex.assert_equal_shape([log_probs, log_rewards, mask])
    p = jax.nn.softmax(log_probs, where=mask)
    r = jax.nn.softmax(log_rewards, where=mask)
    return 0.5 * masked_sum(jnp.abs(p - r), mask)


def first_occurrence_mask(states: TEnvState) -> Bool[Array, " n_states"]:
    """Mark the first occurrence of each distinct state in a batch.

    Two states are equal when every leaf of their pytrees is equal. The comparison
    is done row by row, so memory stays linear in the batch size.

    Args:
        states: Batched states with leading dimension `n_states`.

    Returns:
        Boolean mask of shape `[n_states]`; `True` keeps the element.
    """
    leaves = [x.reshape(x.shape[0], -1) for x in jax.tree.leaves(states)]
    n_states = leaves[0].shape[0]

    def equal_to(i):
        eq = jnp.ones((n_states,), dtype=jnp.bool_)
        for leaf in leaves:
            eq &= jnp.all(leaf == leaf[i], axis=-1)
        return eq

    is_equal = jax.lax.map(equal_to, jnp.arange(n_states))  # [n, n]
    has_earlier_copy = jnp.any(jnp.tril(is_equal, k=-1), axis=-1)
    return jnp.logical_not(has_earlier_copy)


@chex.dataclass
class FCSMetricsState(MetricsState):
    """State of the FCS metric.

    Attributes:
        tv_per_subset: Restricted TV distance for each sampled subset, shape
            `[n_subsets]`. FCS is its mean.
    """

    tv_per_subset: Float[Array, " n_subsets"]


@chex.dataclass
class TestFCSMetricsState(FCSMetricsState):
    """State of the test-set FCS metric.

    Attributes:
        test_set: Fixed pool of terminal states from which subsets are drawn.
    """

    test_set: TEnvState


class BaseFCSMetricsModule(BaseMetricsModule):
    r"""Abstract base for the Flow Consistency in Sub-graphs (FCS) metric.

    On each evaluation, `n_subsets` subsets of at most `subset_size` terminal states
    are drawn (how is defined by the subclass), $\log p_T(x)$ is estimated for every
    element with `n_rounds` backward trajectories, and the TV distance between the
    restrictions of $p_T$ and $R$ to each subset is averaged.

    Reported metrics:
        - `fcs`: Monte Carlo estimate of FCS, in `[0, 1]` (lower is better).

    Attributes:
        env: Environment instance.
        reward_module: Reward module used to score the subset elements.
        bwd_policy_fn: Backward policy whose info dict exposes both
            `forward_logits` and `backward_logits`.
        n_subsets: Number of subsets $S$ averaged per evaluation.
        subset_size: Maximum subset size $\beta$ (the paper uses the training batch size).
        n_rounds: Backward trajectories per terminal state for estimating $p_T$.
        batch_size: Backward rollouts run in parallel; bounds memory use.
        deduplicate: Whether repeated states within a subset are counted once, so that
            $S$ is a set rather than a multiset (as in the paper).
    """

    def __init__(
        self,
        env: TEnvironment,
        reward_module: TRewardModule,
        bwd_policy_fn: TPolicyFn,
        subset_size: int,
        n_subsets: int = 32,
        n_rounds: int = 10,
        batch_size: int | None = None,
        deduplicate: bool = True,
    ):
        r"""Initialize the FCS metric module.

        Args:
            env: Environment instance.
            reward_module: Reward module used to score the subset elements.
            bwd_policy_fn: Backward policy function for estimating $p_T$.
            subset_size: Maximum subset size $\beta \geq 2$.
            n_subsets: Number of subsets averaged per evaluation.
            n_rounds: Backward trajectories per terminal state.
            batch_size: Backward rollouts run in parallel. Defaults to `subset_size`.
            deduplicate: Count repeated states within a subset once.
        """
        if subset_size < 2:
            raise ValueError(f"subset_size must be at least 2, got {subset_size}")
        if n_subsets < 1:
            raise ValueError(f"n_subsets must be positive, got {n_subsets}")
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be positive, got {n_rounds}")
        self.env = env
        self.reward_module = reward_module
        self.bwd_policy_fn = bwd_policy_fn
        self.subset_size = subset_size
        self.n_subsets = n_subsets
        self.n_rounds = n_rounds
        self.batch_size = batch_size if batch_size is not None else subset_size
        self.deduplicate = deduplicate

    UpdateArgs = EmptyUpdateArgs

    def update(
        self,
        metrics_state: FCSMetricsState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> FCSMetricsState:
        """No-op: FCS is computed from scratch at each evaluation."""
        return metrics_state

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for computing FCS.

        Attributes:
            policy_params: Current policy parameters, used for the backward rollouts
                (and, on-policy, for sampling the subsets).
            env_params: Environment parameters.
            reward_params: Current reward parameters.
        """

        policy_params: TPolicyParams
        env_params: TEnvParams
        reward_params: TRewardParams

    def process(
        self,
        metrics_state: FCSMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> FCSMetricsState:
        """Sample subsets and compute the restricted TV distance for each."""
        subset_key, marginal_key = jax.random.split(rng_key)
        subset_states = self._sample_subsets(metrics_state, subset_key, args)
        tv_per_subset = self.compute_tv_per_subset(
            rng_key=marginal_key,
            subset_states=subset_states,
            policy_params=args.policy_params,
            env_params=args.env_params,
            reward_params=args.reward_params,
        )
        return metrics_state.replace(tv_per_subset=tv_per_subset)

    def get(self, metrics_state: FCSMetricsState) -> dict[str, Any]:
        """Return the FCS estimate (mean restricted TV over subsets)."""
        return {"fcs": jnp.mean(metrics_state.tv_per_subset)}

    def compute_tv_per_subset(
        self,
        rng_key: chex.PRNGKey,
        subset_states: TEnvState,
        policy_params: TPolicyParams,
        env_params: TEnvParams,
        reward_params: TRewardParams,
    ) -> Float[Array, " n_subsets"]:
        """Restricted TV distance for each of the given subsets.

        Args:
            rng_key: PRNG key for the backward rollouts.
            subset_states: Terminal states of shape `[n_subsets, subset_size, ...]`.
            policy_params: Parameters for the backward policy.
            env_params: Environment parameters.
            reward_params: Reward parameters.

        Returns:
            TV distance between $p_T^{(S)}$ and $R^{(S)}$ for each subset.
        """
        batch_shape = subset_states.is_pad.shape
        chex.assert_rank(subset_states.is_pad, 2)
        if self.deduplicate:
            mask = jax.vmap(first_occurrence_mask)(subset_states)
        else:
            mask = jnp.ones(batch_shape, dtype=jnp.bool_)

        flat_states = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), subset_states)
        log_probs = estimate_marginal_log_probs(
            rng_key=rng_key,
            terminal_states=flat_states,
            bwd_policy_fn=self.bwd_policy_fn,
            policy_params=policy_params,
            env=self.env,
            env_params=env_params,
            n_rounds=self.n_rounds,
            batch_size=self.batch_size,
        ).reshape(batch_shape)
        log_rewards = jax.vmap(self.reward_module.log_reward, in_axes=(0, None))(
            flat_states, reward_params
        ).reshape(batch_shape)
        return jax.vmap(subset_total_variation)(log_probs, log_rewards, mask)

    @abstractmethod
    def _sample_subsets(
        self,
        metrics_state: FCSMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> TEnvState:
        """Draw terminal states of shape `[n_subsets, subset_size, ...]`."""
        raise NotImplementedError


class OnPolicyFCSMetricsModule(BaseFCSMetricsModule):
    r"""FCS with subsets drawn from the current forward policy.

    Each subset is the set of terminal states of a fresh batch of `subset_size`
    forward rollouts, so $P_S$ is the distribution over at-most-$\beta$-sized subsets
    induced by the sampler. This is the implementation used by Silva et al. (2025),
    which weights subsets towards the regions where the GFlowNet puts its mass.
    """

    def __init__(
        self,
        env: TEnvironment,
        reward_module: TRewardModule,
        fwd_policy_fn: TPolicyFn,
        bwd_policy_fn: TPolicyFn,
        subset_size: int,
        n_subsets: int = 32,
        n_rounds: int = 10,
        batch_size: int | None = None,
        deduplicate: bool = True,
    ):
        r"""Initialize the on-policy FCS metric module.

        Args:
            env: Environment instance.
            reward_module: Reward module used to score the subset elements.
            fwd_policy_fn: Forward policy used to sample the subsets.
            bwd_policy_fn: Backward policy function for estimating $p_T$.
            subset_size: Number of forward rollouts per subset ($\beta \geq 2$).
            n_subsets: Number of subsets averaged per evaluation.
            n_rounds: Backward trajectories per terminal state.
            batch_size: Backward rollouts run in parallel. Defaults to `subset_size`.
            deduplicate: Count repeated states within a subset once.
        """
        super().__init__(
            env=env,
            reward_module=reward_module,
            bwd_policy_fn=bwd_policy_fn,
            subset_size=subset_size,
            n_subsets=n_subsets,
            n_rounds=n_rounds,
            batch_size=batch_size,
            deduplicate=deduplicate,
        )
        self.fwd_policy_fn = fwd_policy_fn

    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs | None = None) -> FCSMetricsState:
        """Initialize the state with zero placeholders."""
        return FCSMetricsState(tv_per_subset=jnp.zeros((self.n_subsets,)))

    def _sample_subsets(
        self,
        metrics_state: FCSMetricsState,
        rng_key: chex.PRNGKey,
        args: BaseFCSMetricsModule.ProcessArgs,
    ) -> TEnvState:
        """Run `n_subsets` batches of `subset_size` forward rollouts, one batch at a time."""

        def sample_subset(key):
            rollout_keys = jax.random.split(key, self.subset_size)
            _, final_states, _ = jax.vmap(
                lambda rng: forward_rollout(
                    rng, self.fwd_policy_fn, args.policy_params, self.env, args.env_params
                )
            )(rollout_keys)
            return final_states

        return jax.lax.map(sample_subset, jax.random.split(rng_key, self.n_subsets))


class TestFCSMetricsModule(BaseFCSMetricsModule):
    r"""FCS with subsets drawn uniformly from a fixed test set of terminal states.

    Each subset is a uniformly random `subset_size`-subset (without replacement) of
    the test set, so $P_S$ stays fixed across training and scores are comparable
    between evaluations. With `subset_size` equal to the test-set size, FCS equals
    the TV distance between $p_T$ and $R$ restricted to the test set.
    """

    __test__ = False  # Not a pytest test class despite the `Test` prefix.

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the test-set FCS module.

        Attributes:
            test_set: Batched terminal states to draw subsets from; must contain at
                least `subset_size` states.
        """

        test_set: TEnvState

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> TestFCSMetricsState:
        """Store the test set and zero placeholders."""
        test_set_size = args.test_set.is_pad.shape[0]
        if test_set_size < self.subset_size:
            raise ValueError(
                f"The test set has {test_set_size} states, fewer than"
                f" subset_size={self.subset_size}"
            )
        return TestFCSMetricsState(
            tv_per_subset=jnp.zeros((self.n_subsets,)),
            test_set=args.test_set,
        )

    def _sample_subsets(
        self,
        metrics_state: TestFCSMetricsState,
        rng_key: chex.PRNGKey,
        args: BaseFCSMetricsModule.ProcessArgs,
    ) -> TEnvState:
        """Draw `n_subsets` uniformly random subsets of the test set."""
        test_set_size = metrics_state.test_set.is_pad.shape[0]
        indices = jax.vmap(
            lambda key: jax.random.choice(
                key, test_set_size, shape=(self.subset_size,), replace=False
            )
        )(jax.random.split(rng_key, self.n_subsets))  # [n_subsets, subset_size]
        return jax.tree.map(lambda x: x[indices], metrics_state.test_set)
