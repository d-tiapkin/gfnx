import jax
import matplotlib
from matplotlib.animation import FuncAnimation

from gfnx.base import BaseVecEnvironment, BaseEnvParams, BaseEnvState

from .sequence import SequenceEnvironment, SequenceRenderer
from .hypergrid import HypergridEnvironment, HypergridRenderer
from ..utils import TrajectoryData, split_traj_to_transitions


class Visualizer:
    """A visualization utility for rendering GFlowNet environments and trajectories.

    The Visualizer class provides methods to render static states, create animations
    of trajectories, and visualize complete trajectory sequences. It automatically
    selects the appropriate rendering backend based on the environment type.

    Currently supports:
        - SequenceEnvironment: Uses SequenceRenderer backend
        - HypergridEnvironment: Uses HypergridRenderer backend

    Args:
        env: The vectorized environment to visualize
        env_params: Environment parameters
        dpi: Dots per pixel - controls image resolution and quality. Higher values
             produce sharper images but larger file sizes. Common values:
             - 1-2: Low resolution, fast rendering, small files
             - 3-4: Medium resolution, good balance
             - 5+: High resolution, slower rendering, large files

    Raises:
        NotImplementedError: If the environment type is not supported

    Example:
        >>> visualizer = Visualizer(env, env_params)
        >>> visualizer.render(state, "state.png")
        >>> visualizer.animate(trajectory, "animation.mp4")
    """

    def __init__(
        self, env: BaseVecEnvironment, env_params: BaseEnvParams, dpi: float = 200.0
    ):
        """Initialize the Visualizer with the given environment and parameters.

        Args:
            env: The vectorized environment to visualize
            env_params: Environment parameters
            dpi: Dots per inches - image resolution multiplier. Controls the pixel
                 density of rendered images. Higher DPI = more pixels = sharper
                 images but slower rendering and larger files (default: 200.0)

        Raises:
            NotImplementedError: If the environment type is not supported
        """
        matplotlib.use("Agg")

        if issubclass(type(env), SequenceEnvironment):
            self.backend = SequenceRenderer(env, env_params, dpi=dpi)
        elif issubclass(type(env), HypergridEnvironment):
            self.backend = HypergridRenderer(env, env_params, dpi=dpi)
        else:
            raise NotImplementedError("Unsupported environment type")

    def render(self, state: BaseEnvState, save_path: str):
        """Render a single environment state to a static image.

        Args:
            state: The environment state to render
            save_path: Path where the rendered image will be saved
        """
        self.backend.init_state(state)
        self.backend.figure.savefig(save_path)

    def animate(self, trajectory: TrajectoryData, save_path: str, interval: int = 250):
        """Create an animated visualization of a trajectory sequence.

        Generates a frame-by-frame animation showing the progression through
        states in the trajectory. Each frame shows the transition from one
        state to the next based on the actions taken.

        Args:
            trajectory: Trajectory data containing states, actions, and transitions
            save_path: Path where the animation will be saved (e.g., .mp4, .gif)
            interval: Delay between frames in milliseconds (default: 250)

        Note:
            The trajectory is expected to have shape [B, T, ...]
            where B is batch size and T is trajectory length.
        """
        batch_size, traj_len = trajectory.action.shape
        transitions = jax.tree.map(
            lambda x: x.reshape((batch_size, traj_len - 1) + tuple(x.shape[1:])),
            split_traj_to_transitions(trajectory),
        )  # Reshape to [B x T x ...]

        self.backend.init_state(
            jax.tree.map(lambda x: x[:, 0, ...], transitions.state)
        )  # Initial state

        def update_animation_frame(step):
            if step == 0:
                return
            transition = jax.tree.map(lambda x: x[:, step - 1, ...], transitions)
            state, next_state, action = transition.state, transition.next_state, transition.action
            self.backend.transition(state, next_state, action)

        animation = FuncAnimation(
            self.backend.figure,
            update_animation_frame,
            frames=traj_len,
            interval=interval,
            blit=False,
        )
        animation.save(save_path)

    def render_trajectory(self, trajectory: TrajectoryData, save_path: str):
        """Render a complete trajectory sequence to a single static image.

        Processes all transitions in the trajectory and renders the final
        cumulative state to a static image. This is useful for visualizing
        the complete path taken through the environment.

        Args:
            trajectory: Trajectory data containing states, actions, and transitions
            save_path: Path where the rendered image will be saved

        Note:
            Unlike animate(), this method applies all transitions sequentially
            and saves only the final visualization state as a static image.
            The trajectory is expected to have shape [B, T, ...] where B is
            batch size and T is trajectory length.
        """
        batch_size, traj_len = trajectory.action.shape
        transitions = jax.tree.map(
            lambda x: x.reshape((batch_size, traj_len - 1) + tuple(x.shape[1:])),
            split_traj_to_transitions(trajectory),
        )  # Reshape to [B x T x ...]

        self.backend.init_state(
            jax.tree.map(lambda x: x[:, 0, ...], transitions.state)
        )  # Initial state
        for step in range(traj_len - 1):
            transition = jax.tree.map(lambda x: x[:, step, ...], transitions)
            state, next_state, action = transition.state, transition.next_state, transition.action
            self.backend.transition(state, next_state, action)

        self.backend.figure.savefig(save_path)
