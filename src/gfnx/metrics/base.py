import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

import chex
import jax


class MetricsState:
    """Base state for metric computation.

    This is an abstract base class that serves as a pure data container for metric states.
    Metric states are designed to only store the necessary data and intermediate values
    required for computing metrics during training and evaluation. They should not
    contain any methods or processing logic.

    All processing, computation, and transformation logic should be implemented in the
    corresponding BaseMetricsModule subclasses, not in the state objects themselves.

    Subclasses should define specific data fields (typically using @chex.dataclass)
    needed for their metric computation requirements, but no methods.
    """

    pass


class BaseInitArgs(ABC):
    """
    Base class for argument containers passed as `args` to the `init` method
    of metric modules.
    """

    pass


class BaseUpdateArgs(ABC):
    """
    Base class for argument containers passed as `args` to the `update` method
    of metric modules.
    """

    pass


class BaseProcessArgs(ABC):
    """
    Base class for argument containers passed as `args` to the `process` method
    of metric modules.
    """

    pass


# Some helper classes for empty arguments cases


@chex.dataclass
class EmptyInitArgs(BaseInitArgs):
    """Empty initialization arguments for metrics that do not require any parameters."""

    pass


@chex.dataclass
class EmptyUpdateArgs(BaseUpdateArgs):
    """Empty update arguments for metrics that do not require any parameters."""

    pass


@chex.dataclass
class EmptyProcessArgs(BaseProcessArgs):
    """Empty process arguments for metrics that do not require any parameters."""

    pass


TMetricsState = TypeVar("TMetricsState", bound=MetricsState)
TInitArgs = TypeVar("TInitArgs", bound=BaseInitArgs)
TUpdateArgs = TypeVar("TUpdateArgs", bound=BaseUpdateArgs)
TProcessArgs = TypeVar("TProcessArgs", bound=BaseProcessArgs)


class BaseMetricsModule(ABC, Generic[TInitArgs, TUpdateArgs, TProcessArgs, TMetricsState]):
    """Environment-agnostic base metric module.

    This abstract base class defines the interface for all metric modules in the system.
    Metric modules are responsible for tracking, computing, and reporting metrics during
    training and evaluation phases. They maintain their own state and can be composed
    together using the MultiMetricsModule.

    The lifecycle of a metric module follows this pattern:
    1. init() - Initialize the metric state with required parameters
    2. update() - Update state with new data points during training/evaluation
    3. process() - Apply any final transformations before metric computation
    4. get() - Retrieve the computed metrics as a dictionary

    In addition to implementing the abstract methods, each subclass must define the
    following inner classes to specify argument types:
      - InitArgs, inheriting from BaseInitArgs
      - UpdateArgs, inheriting from BaseUpdateArgs
      - ProcessArgs, inheriting from BaseProcessArgs

    Subclasses that do not provide these classes or methods will raise errors via
    __init_subclass__ validation.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # skip verification if the class is still abstract
        if inspect.isabstract(cls):
            return

        # 1. Must define InitArgs, UpdateArgs, ProcessArgs
        if not hasattr(cls, "InitArgs"):
            raise TypeError(f"{cls.__name__} must define an InitArgs class")
        if not hasattr(cls, "UpdateArgs"):
            raise TypeError(f"{cls.__name__} must define an UpdateArgs class")
        if not hasattr(cls, "ProcessArgs"):
            raise TypeError(f"{cls.__name__} must define a ProcessArgs class")

        # 2. Must inherit from BaseInitArgs
        if not issubclass(cls.InitArgs, BaseInitArgs):
            raise TypeError(f"{cls.__name__}.InitArgs must inherit from BaseInitArgs")
        if not issubclass(cls.UpdateArgs, BaseUpdateArgs):
            raise TypeError(f"{cls.__name__}.UpdateArgs must inherit from BaseUpdateArgs")
        if not issubclass(cls.ProcessArgs, BaseProcessArgs):
            raise TypeError(f"{cls.__name__}.ProcessArgs must inherit from BaseProcessArgs")

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey, args: TInitArgs | None = None) -> TMetricsState:
        """Initialize metric state.

        Creates and returns the initial state required for metric computation.
        This method is called once at the beginning of training or evaluation
        to set up any necessary data structures, counters, or buffers.

        Args:
            rng_key: JAX PRNG key for any random initialization required
            args: Optional InitArgs object containing metric-specific init parameters

        Returns:
            MetricsState: Initialized state object for this metric

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self, metrics_state: TMetricsState, rng_key: chex.PRNGKey, args: TUpdateArgs | None = None
    ) -> TMetricsState:
        """Update metric state with new data.

        Updates the metric state with new data points collected during training
        or evaluation. This method is called repeatedly as new data becomes available
        and should accumulate or process the information needed for final metric
        computation.

        Args:
            metrics_state: Current state of the metric
            rng_key: JAX PRNG key for any random operations during update
            args: Optional UpdateArgs object containing metric-specific update data

        Returns:
            MetricsState: Updated state object with new data incorporated

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def process(
        self, metrics_state: TMetricsState, rng_key: chex.PRNGKey, args: TProcessArgs | None = None
    ) -> TMetricsState:
        """Process metric state to compute metrics and perform final transformations.

        This method is called exactly to compute metrics before getting their results
        and to perform final transformations during the evaluation period. It prepares
        the accumulated data for final metric computation by applying any necessary
        calculations, normalizations, or statistical operations.

        Args:
            metrics_state: Current state of the metric after all updates
            rng_key: JAX PRNG key for any random operations during processing
            args: Optional ProcessArgs object containing metric-specific processing parameters

        Returns:
            MetricsState: Processed state ready for metric retrieval via get()

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, metrics_state: TMetricsState) -> Dict[str, Any]:
        """Get computed metrics from the current state.

        Computes and returns the final metrics based on the current state.
        This method should extract meaningful metrics from the accumulated
        data and return them in a standardized dictionary format.

        Args:
            metrics_state: Current processed state of the metric

        Returns:
            Dict[str, Any]: Dictionary containing computed metrics with
                          descriptive keys and their corresponding values

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


@chex.dataclass
class MultiMetricsState(MetricsState):
    """State container for multiple metrics.

    This class extends MetricsState to hold the states of multiple individual
    metric modules. It provides a unified interface for managing the states
    of different metrics that need to be computed together.

    Attributes:
        states: Dictionary mapping metric names to their individual MetricsState objects.
               Each key represents a unique metric identifier, and each value is the
               corresponding metric's state object.
    """

    states: Dict[str, MetricsState]


class MultiMetricsModule(BaseMetricsModule):
    """Module for handling multiple metrics in a unified way.

    This class implements the BaseMetricsModule interface to manage multiple
    individual metric modules simultaneously. It provides a convenient way to
    compute multiple metrics together while maintaining the same interface
    as single metric modules.

    The MultiMetricsModule coordinates the lifecycle of all contained metrics,
    calling their respective init, update, process, and get methods in sequence.
    Final metrics are returned with prefixed names to avoid conflicts between
    different metric modules.

    Attributes:
        metrics: Dictionary of metric modules indexed by their names
        _supported_metrics: Internal mapping of metric names to their get methods
    """

    def __init__(self, metrics: Dict[str, BaseMetricsModule]):
        """Initialize the MultiMetricsModule with a collection of metrics.

        Args:
            metrics: Dictionary mapping metric names to their corresponding
                    BaseMetricsModule instances. Names should be unique and
                    descriptive as they will be used as prefixes in the final
                    metric output.
        """
        self.metrics = metrics
        self._supported_metrics = {name: metric.get for name, metric in metrics.items()}

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the MultiMetricsModule."""

        metrics_args: Dict[str, BaseInitArgs]

    def init(self, rng_key: chex.PRNGKey, args: InitArgs | None = None) -> MultiMetricsState:
        """Initialize all contained metrics.

        Calls the init method on each metric module and collects their
        individual states into a MultiMetricsState container.

        Args:
            rng_key: JAX PRNG key passed to each metric's init method
            args: Optional InitArgs object mapping metric names to init args

        Returns:
            MultiMetricsState: Container holding all initialized metric states
        """
        if args is None:
            args = self.ProcessArgs(metrics_args={})
        metrics_keys = jax.random.split(rng_key, len(self.metrics))
        dict_metrics_keys = dict(zip(self.metrics.keys(), metrics_keys))
        states = {
            name: metric.init(rng_key=dict_metrics_keys[name], args=args.metrics_args.get(name))
            for name, metric in self.metrics.items()
        }
        return MultiMetricsState(states=states)

    @chex.dataclass
    class UpdateArgs(BaseUpdateArgs):
        """Arguments for updating the MultiMetricsModule."""

        metrics_args: Dict[str, Any]

    def update(
        self,
        metrics_state: MultiMetricsState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> MultiMetricsState:
        """Update all contained metrics with new data.

        Calls the update method on each metric module with their corresponding
        state and the provided data, then returns a new MultiMetricsState with
        all updated states.

        Args:
            metrics_state: Current MultiMetricsState containing all metric states
            rng_key: JAX PRNG key passed to each metric's update method
            args: Optional UpdateArgs object mapping metric names to update data

        Returns:
            MultiMetricsState: Container with all updated metric states
        """
        if args is None:
            args = self.ProcessArgs(metrics_args={})
        metrics_keys = jax.random.split(rng_key, len(self.metrics))
        dict_metrics_keys = dict(zip(self.metrics.keys(), metrics_keys))
        updated_states = {
            name: metric.update(
                metrics_state=metrics_state.states[name],
                rng_key=dict_metrics_keys[name],
                args=args.metrics_args.get(name),
            )
            for name, metric in self.metrics.items()
        }
        return metrics_state.replace(states=updated_states)

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the MultiMetricsModule."""

        metrics_args: Dict[str, Any]

    def process(
        self,
        metrics_state: MultiMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs | None = None,
    ) -> MultiMetricsState:
        """Process all metric states to compute metrics and perform final transformations.

        Calls the process method on each metric module with their corresponding
        state to compute metrics and apply final transformations during the
        evaluation period, preparing them for result retrieval.

        Args:
            metrics_state: Current MultiMetricsState containing all metric states
            rng_key: JAX PRNG key passed to each metric's process method
            args: Optional ProcessArgs object mapping metric names to process params

        Returns:
            MultiMetricsState: Container with all processed metric states ready for get()
        """
        if args is None:
            args = self.ProcessArgs(metrics_args={})
        metrics_keys = jax.random.split(rng_key, len(self.metrics))
        dict_metrics_keys = dict(zip(self.metrics.keys(), metrics_keys))
        processed_states = {
            name: metric.process(
                metrics_state=metrics_state.states[name],
                rng_key=dict_metrics_keys[name],
                args=args.metrics_args.get(name),
            )
            for name, metric in self.metrics.items()
        }
        return metrics_state.replace(states=processed_states)

    def get(self, metrics_state: MultiMetricsState) -> Dict[str, Any]:
        """Get computed metrics from all contained modules.

        Collects metrics from all metric modules and returns them with
        prefixed names to avoid conflicts. Each metric's results are
        prefixed with its module name followed by a forward slash.

        Args:
            metrics_state: Current MultiMetricsState containing all processed states

        Returns:
            Dict[str, Any]: Dictionary of all computed metrics with prefixed names.
                          Keys are in the format "{metric_name}/{original_key}".

        Example:
            If a metric module named "accuracy" returns {"score": 0.95},
            the result will include {"accuracy/score": 0.95}.
        """
        results = {}
        for name, state in metrics_state.states.items():
            metrics = self.metrics[name].get(state)
            for key, value in metrics.items():
                results[f"{name}/{key}"] = value
        return results
