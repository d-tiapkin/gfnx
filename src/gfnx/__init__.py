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
    "AMPEnvParams",
    "AMPEnvState",
    "AMPEnvironment",
    "BitseqEnvParams",
    "BitseqEnvState",
    "BitseqEnvironment",
    "BitseqRewardModule",
    "DAGEnvParams",
    "DAGEnvParams",
    "DAGEnvState",
    "DAGEnvState",
    "DAGEnvironment",
    "DAGEnvironment",
    "DAGRewardModule",
    "DAGRewardModule",
    "EasyHypergridRewardModule",
    "EqxProxyAMPRewardModule",
    "EqxProxyGFPRewardModule",
    "GFPEnvParams",
    "GFPEnvState",
    "GFPEnvironment",
    "GeneralHypergridRewardModule",
    "HardHypergridRewardModule",
    "HypergridEnvParams",
    "HypergridEnvState",
    "HypergridEnvironment",
    "IsingEnvParams",
    "IsingEnvState",
    "IsingEnvironment",
    "IsingRewardModule",
    "PhyloTreeEnvParams",
    "PhyloTreeEnvState",
    "PhyloTreeEnvironment",
    "PhyloTreeRewardModule",
    "QM9SmallEnvParams",
    "QM9SmallEnvState",
    "QM9SmallEnvironment",
    "QM9SmallRewardModule",
    "TAction",
    "TBackwardAction",
    "TDone",
    "TEnvParams",
    "TEnvState",
    "TEnvironment",
    "TFBind8EnvParams",
    "TFBind8EnvState",
    "TFBind8Environment",
    "TFBind8RewardModule",
    "TLogReward",
    "TObs",
    "TReward",
    "TRewardModule",
    "TRewardParams",
    "Visualizer",
    "metrics",
    "networks",
    "spaces",
    "utils",
]

# Lazy import of networks since networks are based on Equinox
import importlib


def __getattr__(name):
    if name == "networks":
        return importlib.import_module(f"{__name__}.networks")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
