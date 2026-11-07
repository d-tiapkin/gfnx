"""Hashable, read-only snapshot of a Hydra/OmegaConf config.

The baselines keep their config in a ``TrainState`` NamedTuple that is passed to an
``eqx.filter_jit``-compiled ``train_step``. Non-array leaves of that state become static
arguments, which JAX requires to be hashable. ``omegaconf.DictConfig`` is unhashable as of
omegaconf 2.4 (pulled in by hydra-core 1.4), so the config is frozen before it goes into
the state. Attribute and item access work as on ``DictConfig``
(``config.logging.eval_each``, ``config.logging["eval_each"]``).
"""

from collections.abc import Iterator, Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


class FrozenConfig(Mapping):
    """Immutable, hashable mapping with attribute access to its keys."""

    __slots__ = ("_hash", "_items")

    def __init__(self, items: Mapping[str, Any]):
        object.__setattr__(self, "_items", dict(items))
        object.__setattr__(self, "_hash", hash(tuple(sorted(self._items.items()))))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenConfig is read-only")

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenConfig):
            return NotImplemented
        return self._hash == other._hash and self._items == other._items

    def __repr__(self) -> str:
        return f"FrozenConfig({self._items!r})"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenConfig({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(val) for val in value)
    return value


def freeze_config(cfg: DictConfig) -> FrozenConfig:
    """Resolve ``cfg`` and return it as a nested ``FrozenConfig`` (lists become tuples)."""
    return _freeze(OmegaConf.to_container(cfg, resolve=True))
