# src/config/full_config.py

from dataclasses import dataclass
from typing import Optional, List
from src.config.node_config import NODE_CONFIG
from src.config.pipeline_config import PIPELINE_CONFIG
from src.config.model_config import MODEL_CONFIG

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
    it: int
    nT: int

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

def build_config(node, rate, pipeline, bias_mode):
    node_cfg = NODE_CONFIG[node]
    pipe_cfg = PIPELINE_CONFIG[node, rate, pipeline]
    model_cfg  = MODEL_CONFIG[node, rate, bias_mode]

    return FullConfig(**node_cfg, **pipe_cfg, **model_cfg) 
