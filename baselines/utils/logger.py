import os
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from PIL.Image import Image as PILImage
from PIL.Image import fromarray as pil_fromarray, open as pil_open


class BaseLoggerWrapper(ABC):
    """Abstract base class for logger wrappers."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the logger wrapper."""
        raise NotImplementedError

    @abstractmethod
    def log(self, data: dict, **kwargs):
        """Log data to the backend."""
        raise NotImplementedError

    @abstractmethod
    def finish(self, **kwargs):
        """Finish logging and cleanup resources."""
        raise NotImplementedError

    @abstractmethod
    def Image(self, *args, **kwargs):
        """Format an image for the backend."""
        raise NotImplementedError


class TrackIOWrapper(BaseLoggerWrapper):
    """Logger wrapper for TrackIO experiment tracking."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        space_id: str | None = None,
        dataset_id: str | None = None,
        config: dict | None = None,
        resume: str = "never",
        **kwargs,
    ):
        """Initialize TrackIO logging session."""
        import trackio

        self.writer = trackio.init(
            project,
            name=name,
            space_id=space_id,
            dataset_id=dataset_id,
            config=config,
            resume=resume,
        )

    def log(self, data: dict, **kwargs):
        """Log data to TrackIO."""
        self.writer.log(data)

    def finish(self, **kwargs):
        """Finish TrackIO session."""
        self.writer.finish()

    def Image(self, *args, **kwargs):
        """TrackIO does not support image logging."""
        raise NotImplementedError("TrackIO does not support image logging directly.")


class WandBWrapper(BaseLoggerWrapper):
    """Logger wrapper for Weights & Biases."""

    def __init__(self, *args, **kwargs):
        """Initialize W&B logging session."""
        import wandb

        wandb.init(*args, **kwargs)
        self.writer = wandb

    def log(self, data: dict, step: int | None = None, commit: bool | None = None, **kwargs):
        """Log data to W&B."""
        self.writer.log(data, step=step, commit=commit)

    def finish(self, exit_code: int | None = None, **kwargs):
        """Finish W&B session."""
        self.writer.finish(exit_code=exit_code)

    def Image(self, *args, **kwargs):
        """Create W&B Image object."""
        return self.writer.Image(*args, **kwargs)


class Writer:
    """Unified experiment logging interface supporting multiple backends."""

    SUPPORTED_TYPES = ("wandb", "trackio", None)

    def init(
        self,
        writer_type: str | None = None,
        save_locally: bool = False,
        log_dir: str | None = None,
        config: dict | None = None,
        **kwargs,
    ):
        """Initialize Writer with backend and local storage configuration.

        Args:
            writer_type: Backend type ("wandb", "trackio", or None)
            save_locally: Whether to save logs locally
            log_dir: Local log directory (required if save_locally=True)
            config: Configuration dict to save and pass to backend
            **kwargs: Backend-specific initialization arguments
        """
        assert writer_type in self.SUPPORTED_TYPES, (
            f"Writer type {writer_type} is not supported. "
            f"Supported types are: {self.SUPPORTED_TYPES}."
        )
        assert not (save_locally and log_dir is None), (
            "If save_locally is True, log_dir must be specified."
        )

        if writer_type == "wandb":
            self.backend = WandBWrapper(config=config, **kwargs)
        elif writer_type == "trackio":
            self.backend = TrackIOWrapper(config=config, **kwargs)
        elif writer_type is None:
            self.backend = None

        self.log_dir = log_dir if save_locally else None
        self.image_dir = os.path.join(self.log_dir, "images/")
        os.makedirs(self.image_dir, exist_ok=True)

        self._step = 0
        self._image_counter = 0

        if self.log_dir:
            self._save_json("config.json", config)

    def log(self, data: dict, **kwargs):
        """Log data to both local storage and backend if configured.

        Args:
            data: Dictionary of metrics and values to log
            **kwargs: Additional arguments for backend logging
        """
        if self.log_dir:
            self._save_logs(data)

        if self.backend:
            self.backend.log(data, **kwargs)

    def Image(self, data_or_path: str | Path | np.ndarray | PILImage, **kwargs):
        """Create image object for logging from various input formats.

        Args:
            data_or_path: Image data (file path, numpy array, or PIL Image)
            **kwargs: Additional arguments for backend Image creation

        Returns:
            Backend-specific image object or PIL Image if no backend
        """
        image = self._load_image(data_or_path)
        return self.backend.Image(image, **kwargs) if self.backend else image

    def finish(self, *args, **kwargs):
        """Finish logging session and cleanup resources."""
        if self.backend:
            self.backend.finish(*args, **kwargs)

    def _save_logs(self, data: dict):
        """Save data to local JSONL file with step counter."""
        save_dict = {}
        for key, value in data.items():
            if jnp.isscalar(value):
                save_dict[key] = float(value)
            elif isinstance(value, PILImage):
                save_dict[key] = self._save_image(value)
            elif hasattr(value, "image"):
                save_dict[key] = self._save_image(value.image)

        save_dict["step"] = self._step
        self._step += 1

        log_file = os.path.join(self.log_dir, "log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False)
            f.write("\n")

    def _save_image(self, image: PILImage):
        """Save PIL Image to local directory with sequential naming.

        Args:
            image: PIL Image to save

        Returns:
            str: Path to saved image file
        """

        image_path = os.path.join(self.image_dir, f"image_{self._image_counter:04d}.png")
        self._image_counter += 1
        image.save(image_path)

        return image_path
    
    def _save_json(self, name: str, data: dict):
        """Save dictionary as JSON file in log directory.

        Args:
            name: Filename for JSON file
            data: Dictionary to save
        """

        json_path = os.path.join(self.log_dir, name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _load_image(self, data_or_path: str | Path | np.ndarray | PILImage) -> PILImage:
        """Load image from file path, numpy array, or PIL Image.

        Args:
            data_or_path: Image data in supported format

        Returns:
            PIL Image object

        Raises:
            ValueError: If input type is not supported
        """

        if isinstance(data_or_path, (str, Path)):
            image = pil_open(data_or_path)
        elif isinstance(data_or_path, np.ndarray):
            image = pil_fromarray(data_or_path.astype(np.uint8))
        elif isinstance(data_or_path, PILImage):
            image = data_or_path
        else:
            raise ValueError("Unsupported image data type.")

        return image
