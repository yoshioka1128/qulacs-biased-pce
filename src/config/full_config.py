# src/config/full_config.py

from dataclasses import dataclass
from typing import Optional, List
from src.config.node_config import NODE_CONFIG
from src.config.problem_config import PROBLEM_CONFIG
from src.config.experiment_config import EXPERIMENT_CONFIG

@dataclass
class FullConfig:
    n_qubits: int
    k: int
    alphasc: float
    beta: float
    calpha: float
    bound: float
    iinit: int
    imax0: int
    nseed: int

    # optional
    ninit: int = 5
    strbp: str = "_backprop"
    depth: int = 5
    subcounts: int = 4001
    method: str = "BFGS"
    iseed: int = 42
    type_ansatz: str = "all2all"
    chbetaiinit: Optional[List[int]] = None
    betas: Optional[List[float]] = None

def build_config(node, rate, mode, model):
    node_cfg = NODE_CONFIG[node]
    prob_cfg = PROBLEM_CONFIG[node, rate, mode]
    exp_cfg  = EXPERIMENT_CONFIG[node, rate, mode, model]

    return FullConfig(**node_cfg, **prob_cfg, **exp_cfg) #, **prob_cfg, **exp_cfg)
