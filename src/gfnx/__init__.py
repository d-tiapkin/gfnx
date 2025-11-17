from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import networks

from . import metrics, spaces, utils
from .base import (
    TAction,
    TBackwardAction,
    TDone,
    TEnvironment,
    TEnvParams,
    TEnvState,
    TLogReward,
    TObs,
    TReward,
    TRewardModule,
    TRewardParams,
)
from .environment import (
    AMPEnvironment,
    AMPEnvParams,
    AMPEnvState,
    BitseqEnvironment,
    BitseqEnvParams,
    BitseqEnvState,
    DAGEnvironment,
    DAGEnvParams,
    DAGEnvState,
    GFPEnvironment,
    GFPEnvParams,
    GFPEnvState,
    HypergridEnvironment,
    HypergridEnvParams,
    HypergridEnvState,
)
from .reward import (
    BitseqRewardModule,
    DAGRewardModule,
    EasyHypergridRewardModule,
    EqxProxyAMPRewardModule,
    EqxProxyGFPRewardModule,
    GeneralHypergridRewardModule,
    HardHypergridRewardModule,
)

__all__ = [
    "metrics",
    "networks",
    "spaces",
    "utils",
    "AMPEnvironment",
    "AMPEnvParams",
    "AMPEnvState",
    "BitseqEnvironment",
    "BitseqEnvParams",
    "BitseqEnvState",
    "BitseqRewardModule",
    "EasyHypergridRewardModule",
    "EqxProxyAMPRewardModule",
    "EqxProxyGFPRewardModule",
    "GFPEnvironment",
    "GFPEnvParams",
    "GFPEnvState",
    "GeneralHypergridRewardModule",
    "HardHypergridRewardModule",
    "HypergridEnvironment",
    "HypergridEnvParams",
    "HypergridEnvState",
    "TAction",
    "TBackwardAction",
    "TDone",
    "TEnvParams",
    "TEnvState",
    "TEnvironment",
    "TLogReward",
    "TObs",
    "TReward",
    "TRewardModule",
    "TRewardParams",
    "DAGEnvironment",
    "DAGEnvState",
    "DAGEnvParams",
    "DAGRewardModule",
]

# Lazy import of networks since networks are based on Equinox
import importlib


def __getattr__(name):
    if name == "networks":
        return importlib.import_module(f"{__name__}.networks")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
