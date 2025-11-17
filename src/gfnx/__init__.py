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
    IsingEnvironment,
    IsingEnvParams,
    IsingEnvState,
    PhyloTreeEnvironment,
    PhyloTreeEnvParams,
    PhyloTreeEnvState,
    QM9SmallEnvironment,
    QM9SmallEnvParams,
    QM9SmallEnvState,
    TFBind8Environment,
    TFBind8EnvParams,
    TFBind8EnvState,
)
from .reward import (
    BitseqRewardModule,
    DAGRewardModule,
    EasyHypergridRewardModule,
    EqxProxyAMPRewardModule,
    EqxProxyGFPRewardModule,
    GeneralHypergridRewardModule,
    HardHypergridRewardModule,
    IsingRewardModule,
    PhyloTreeRewardModule,
    QM9SmallRewardModule,
    TFBind8RewardModule,
)
from .visualize import Visualizer

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
    "PhyloTreeEnvironment",
    "PhyloTreeEnvParams",
    "PhyloTreeEnvState",
    "PhyloTreeRewardModule",
    "TAction",
    "TFBind8Environment",
    "TFBind8EnvParams",
    "TFBind8EnvState",
    "TFBind8RewardModule",
    "QM9SmallEnvironment",
    "QM9SmallEnvParams",
    "QM9SmallEnvState",
    "QM9SmallRewardModule",
    "DAGEnvironment",
    "DAGEnvState",
    "DAGEnvParams",
    "DAGRewardModule",
    "IsingEnvironment",
    "IsingEnvState",
    "IsingEnvParams",
    "IsingRewardModule",
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
    "Visualizer",
]

# Lazy import of networks since networks are based on Equinox
import importlib


def __getattr__(name):
    if name == "networks":
        return importlib.import_module(f"{__name__}.networks")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
