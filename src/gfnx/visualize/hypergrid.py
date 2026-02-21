from typing import ClassVar

import jax.numpy as jnp
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ..base import BaseRenderer, TAction
from ..environment.hypergrid import EnvParams, EnvState, HypergridEnvironment


class HypergridRenderer(BaseRenderer[EnvState]):
    POINTS_PER_INCH = 72
    CELL_WIDTH = 32 / 100
    BORDER_WIDTH = 1 / 100
    TRAIL_WIDTH = 4 / 100

    TOKEN_PATTERNS: ClassVar[dict[str, np.ndarray]] = {
        "circle": np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        ),
    }

    def __init__(
        self,
        env: HypergridEnvironment,
        env_params: EnvParams,
        dpi: float,
        color_map: str = "tab20b",
    ):
        """Initialize the HypergridRenderer."""
        assert env_params.dim in (1, 2), "Only 1D and 2D hypergrids are supported."

        self.env = env
        self.env_params = env_params
        self.dpi = dpi
        self.agent_artists: list[matplotlib.artist.Artist] = []
        self.trajectory_artists: list[matplotlib.artist.Artist] = []

        self.reward_colormap = mcolors.LinearSegmentedColormap.from_list(
            "reward_gradient", [mcolors.to_rgba("lightgray"), mcolors.to_rgba("black")]
        )
        self.agent_colormap = matplotlib.colormaps[color_map]

        self.fig, self.ax = self._create_background()

    def init_state(self, state: EnvState):
        """Create visual representation of the given state."""
        self._clear()

        num_envs = state.state.shape[0]
        for i in range(num_envs):
            if self.env_params.dim == 2:
                position = tuple(state.state[i])
            else:
                position = (0, state.state[i, 0])
            color = self.agent_colormap(i / num_envs)

            artist = self._render_pattern(self.TOKEN_PATTERNS["circle"], color, position)
            self.agent_artists.append(artist)

        self.trajectory_artists = [[] for _ in range(num_envs)]

    def transition(self, state: EnvState, next_state: EnvState, action: TAction):
        """Update visualization for state transition."""
        num_envs = state.state.shape[0]

        for i in range(num_envs):
            if self.env_params.dim == 2:
                cur_position, new_position = tuple(state.state[i]), tuple(next_state.state[i])
            else:
                cur_position, new_position = (0, state.state[i, 0]), (0, next_state.state[i, 0])

            color = self.agent_colormap(i / num_envs)

            if cur_position == new_position:
                continue

            self._remove_artists([self.agent_artists[i]])
            self.agent_artists[i] = self._render_pattern(
                self.TOKEN_PATTERNS["circle"], color, new_position
            )

            trail_artist = self._render_trail(cur_position, new_position, color)
            self.trajectory_artists[i].append(trail_artist)

    @property
    def figure(self) -> matplotlib.figure.Figure:
        """Return the current matplotlib figure."""
        if self.fig is None:
            raise ValueError("No state has been created yet. Call init_state() first.")

        self.ax.set_xlim(0, self.env_params.side * self.CELL_WIDTH)
        self.ax.set_ylim(
            0, (self.env_params.side if self.env_params.dim == 2 else 1) * self.CELL_WIDTH
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        return self.fig

    def _render_trail(
        self, start_pos: tuple[int, int], end_pos: tuple[int, int], color: np.ndarray
    ) -> matplotlib.artist.Artist:
        """Render a trail between two positions using matplotlib lines."""
        start_x = start_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        start_y = start_pos[1] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        end_x = end_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        end_y = end_pos[1] * self.CELL_WIDTH + self.CELL_WIDTH / 2

        line = matplotlib.lines.Line2D(
            [start_x, end_x],
            [start_y, end_y],
            color=color,
            linewidth=self.TRAIL_WIDTH * self.POINTS_PER_INCH,
            zorder=2,
        )
        self.ax.add_line(line)

        return line

    def _render_pattern(
        self, pattern: np.ndarray, color: np.ndarray, position: tuple[int, int]
    ) -> list[matplotlib.artist.Artist]:
        """Render a single pattern at the specified position."""
        x = position[0] * self.CELL_WIDTH + self.BORDER_WIDTH
        y = position[1] * self.CELL_WIDTH + self.BORDER_WIDTH
        size = self.CELL_WIDTH - 2 * self.BORDER_WIDTH

        pattern_rgba = np.zeros((*pattern.shape, len(color)))
        pattern_rgba[pattern == 1] = color
        return self.ax.imshow(
            pattern_rgba,
            extent=[x, x + size, y + size, y],
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            zorder=3,
        )


    def _remove_artists(self, artists: list[matplotlib.artist.Artist]):
        """Remove matplotlib artists from the plot."""
        for artist in artists:
            artist.remove()

    def _create_background(self):
        """Create figure and background for visualization."""
        height = self.env_params.side if self.env_params.dim == 2 else 1
        width = self.env_params.side
        fig, ax = plt.subplots(
            1, 1, figsize=(width * self.CELL_WIDTH, height * self.CELL_WIDTH), dpi=self.dpi
        )

        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                width * self.CELL_WIDTH,
                height * self.CELL_WIDTH,
                facecolor=mcolors.CSS4_COLORS["grey"],
                zorder=0,
            )
        )

        reward_grid = self._get_reward_grid()
        reward_grid /= reward_grid.max() if reward_grid.max() > 0 else 1
        for x in range(width):
            for y in range(height):
                ax.add_patch(
                    patches.Rectangle(
                        (
                            x * self.CELL_WIDTH + self.BORDER_WIDTH,
                            y * self.CELL_WIDTH + self.BORDER_WIDTH,
                        ),
                        self.CELL_WIDTH - 2 * self.BORDER_WIDTH,
                        self.CELL_WIDTH - 2 * self.BORDER_WIDTH,
                        facecolor=self.reward_colormap(reward_grid[y, x]),
                        zorder=1,
                    )
                )

        return fig, ax

    def _get_reward_grid(self) -> np.ndarray:
        """Create a reward grid for rendering."""
        total_states = self.env_params.side**self.env_params.dim
        shape_tuple = tuple([self.env_params.side] * int(self.env_params.dim))
        states = jnp.unravel_index(jnp.arange(total_states), shape_tuple)
        state_array = jnp.stack(states, axis=1)

        dummy_state = EnvState(
            state=state_array,
            is_terminal=jnp.ones((total_states,), dtype=jnp.bool),
            is_initial=jnp.zeros((total_states,), dtype=jnp.bool),
            is_pad=jnp.zeros((total_states,), dtype=jnp.bool),
        )

        reward = self.env.reward_module.reward(dummy_state, self.env_params)
        reward_grid = reward.reshape(shape_tuple)

        return np.array(reward_grid)

    def _clear(self):
        """Clear the renderer."""
        self._remove_artists(self.agent_artists)
        for agent_trails in self.trajectory_artists:
            self._remove_artists(agent_trails)

        self.agent_artists.clear()
        self.trajectory_artists.clear()

    def __del__(self):
        self._clear()
        if self.fig is not None:
            plt.close(self.fig)
