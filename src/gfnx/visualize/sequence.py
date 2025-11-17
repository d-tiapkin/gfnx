import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from typing import Tuple, List

from ..base import BaseRenderer, TAction
from ..environment.sequence import SequenceEnvironment, EnvState, EnvParams


class SequenceRenderer(BaseRenderer[EnvState]):
    PATTERN_SIZE = 8
    CELL_WIDTH = 32 / 100
    BORDER_WIDTH = 1 / 100

    TOKEN_PATTERNS = {
        "char": np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
        ]),
        "bos": np.array([
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ]),
        "eos": np.array([
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ]),
        "pad": np.zeros((PATTERN_SIZE, PATTERN_SIZE), dtype=int),
    }

    def __init__(
        self,
        env: SequenceEnvironment,
        env_params: EnvParams,
        dpi: float,
        color_map: str = "tab20b",
    ):
        """Initialize the SequenceRenderer."""
        self.env = env
        self.env_params = env_params
        self.fig: matplotlib.figure.Figure | None = None
        self.ax: matplotlib.axes.Axes | None = None
        self.dpi = dpi
        self.token_artists: List[matplotlib.artist.Artist] = []

        self.colors = matplotlib.colormaps[color_map](np.linspace(0, 1, env_params.ntoken))

        # Use more visually appealing colors for special tokens
        self.colors[env_params.pad_token] = mcolors.to_rgba("lightgray")
        self.colors[env_params.bos_token] = mcolors.to_rgba("forestgreen")
        self.colors[env_params.eos_token] = mcolors.to_rgba("crimson")

    def init_state(self, state: EnvState):
        """Create visual representation of the given state."""
        self._clear()

        height, width = state.tokens.shape
        self._create_background(height, width)

        self.token_artists = [[[] for _ in range(width)] for _ in range(height)]

        for i in range(height):
            for j in range(width):
                token = state.tokens[i, j]
                pattern = self._get_pattern(token)
                color = self.colors[token]
                position = (j, height - i - 1)  # Reverse Y-axis

                artists = self._render_pattern(pattern, color, position)
                self.token_artists[i][j] = artists

    def transition(self, state: EnvState, next_state: EnvState, action: TAction):
        """Update visualization for state transition."""
        height, width = state.tokens.shape

        for i in range(height):
            for j in range(width):
                cur_token, next_token = state.tokens[i, j], next_state.tokens[i, j]
                if cur_token == next_token:
                    continue

                pattern = self._get_pattern(next_token)
                color = self.colors[next_token]
                position = (j, height - i - 1)  # Reverse Y-axis

                self._remove_token_artists(self.token_artists[i][j])
                self.token_artists[i][j] = self._render_pattern(pattern, color, position)

    @property
    def figure(self) -> matplotlib.figure.Figure:
        """Return the current matplotlib figure."""
        if self.fig is None:
            raise ValueError("No state has been created yet. Call init_state() first.")

        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        return self.fig

    def _render_pattern(
        self, pattern: np.ndarray, color: np.ndarray, position: Tuple[int, int]
    ) -> List[matplotlib.artist.Artist]:
        """Render a single pattern at the specified position."""
        x = position[0] * self.CELL_WIDTH + self.BORDER_WIDTH
        y = position[1] * self.CELL_WIDTH + self.BORDER_WIDTH
        size = self.CELL_WIDTH - 2 * self.BORDER_WIDTH

        background_patch = patches.Rectangle((x, y), size, size, facecolor=color)
        self.ax.add_patch(background_patch)  # render background

        pattern_rgba = np.zeros((*pattern.shape, len(color)))
        pattern_rgba[pattern == 1] = self._get_inverse_color(color)
        pattern_img = self.ax.imshow(
            pattern_rgba,
            extent=[x, x + size, y + size, y],
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            zorder=2,
        )  # render pattern

        return [background_patch, pattern_img]

    def _get_inverse_color(self, base_color: np.ndarray) -> np.ndarray:
        """Generate contrasting color for text overlay."""
        base_rgb = np.array(mcolors.to_rgb(base_color))
        brightness = np.mean(base_rgb)

        if brightness < 0.5:
            inverse_rgb = np.clip(base_rgb * 0.4, 0, 1)
        else:
            inverse_rgb = np.clip(base_rgb * 1.6, 0, 1)

        return np.array(mcolors.to_rgba(inverse_rgb))

    def _get_pattern(self, token_id: int) -> np.ndarray:
        """Get visual pattern for a token."""
        if token_id == self.env_params.bos_token:
            return self.TOKEN_PATTERNS["bos"]
        elif token_id == self.env_params.eos_token:
            return self.TOKEN_PATTERNS["eos"]
        elif token_id == self.env_params.pad_token:
            return self.TOKEN_PATTERNS["pad"]

        return self.TOKEN_PATTERNS["char"]

    def _remove_token_artists(self, artists: List[matplotlib.artist.Artist]):
        """Remove matplotlib artists from the plot."""
        for artist in artists:
            artist.remove()

    def _create_background(self, height: int, width: int):
        """Create figure and background for visualization."""
        self.fig, self.ax = plt.subplots(
            1, 1, figsize=(width * self.CELL_WIDTH, height * self.CELL_WIDTH), dpi=self.dpi
        )
        self.ax.set_xlim(0, width * self.CELL_WIDTH)
        self.ax.set_ylim(0, height * self.CELL_WIDTH)

        self.ax.add_patch(
            patches.Rectangle(
                (0, 0),
                width * self.CELL_WIDTH,
                height * self.CELL_WIDTH,
                facecolor=mcolors.CSS4_COLORS["grey"],
                zorder=0,
            )
        )

    def _clear(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = None, None
        self.token_artists.clear()

    def __del__(self):
        self._clear()
