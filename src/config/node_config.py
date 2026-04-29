# node_config.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NodeConfig:
    n_qubits: int
    k: int
    ninit: int
    iinit: int
    imax0: int
    alphasc: float
    beta: float
    calpha: float
    bound: float
    nseed: int
    strbp: str = "_backprop"
    depth: int = 5
    subcounts: int = 4001
    method: str = "BFGS"
    iseed: int = 42
    type_ansatz: str = "all2all"

    # optional
    chbetaiinit: Optional[List[int]] = None
    betas: Optional[List[float]] = None


# =========================
# Configuration
# =========================
NODE_CONFIG = {
    (18, 0.1, "nobias"): NodeConfig(
        alphasc=2.5, beta=0.1, iinit=4,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        calpha=80, bound=1.0, nseed=24, 
    ),

    (60, 0.1, "nobias"): NodeConfig(
        alphasc=6.0, beta=0.1, iinit=4, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        calpha=6, bound=0.1, nseed=39, 
    ),

    (210, 0.1, "nobias"): NodeConfig(
        alphasc=1.5, beta=0.0, iinit=0, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        calpha=6, bound=0.1, nseed=0, 
    ),

    (756, 0.1, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.2, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.3, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.4, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.5, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.1, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.2, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.3, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.4, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),
    (756, 0.5, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),

    (2772, 0.1, "nobias"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        calpha=2, bound=0.1, nseed=9, 
    ),

    (18, 0.1, "bias_x"): NodeConfig(
        alphasc=2.0, beta=0.1, iinit=3,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        calpha=80, bound=1.0, nseed=24, 
    ),

    (60, 0.1, "bias_x"): NodeConfig(
        alphasc=1.0, beta=0.1, iinit=3, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        calpha=6, bound=0.1, nseed=39, 
    ),

    (210, 0.1, "bias_x"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=3, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        calpha=6, bound=0.1, nseed=0, 
    ),

    (756, 0.1, "bias_x"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=3, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        calpha=2, bound=0.1, nseed=7, 
    ),

    (2772, 0.1, "bias_x"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        calpha=2, bound=0.1, nseed=9, 
    ),

    (18, 0.1, "bias_y"): NodeConfig(
        alphasc=2.5, beta=0.1, iinit=1,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        calpha=80, bound=1.0, nseed=24, 
    ),
    (18, 0.5, "bias_y"): NodeConfig(
        alphasc=2.5, beta=0.1, iinit=1,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        calpha=80, bound=1.0, nseed=24, 
    ),

    (60, 0.1, "bias_y"): NodeConfig(
        alphasc=1.5, beta=0.0, iinit=4, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        calpha=6, bound=0.1, nseed=39, 
    ),
    (60, 0.5, "bias_y"): NodeConfig(
        alphasc=1.5, beta=0.0, iinit=4, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        calpha=6, bound=0.1, nseed=39, 
    ),

    (210, 0.1, "bias_y"): NodeConfig(
        alphasc=0.5, beta=-0.1, iinit=3, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        calpha=6, bound=0.1, nseed=0, 
    ),
    (210, 0.5, "bias_y"): NodeConfig(
        alphasc=0.5, beta=-0.1, iinit=3, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        calpha=6, bound=0.1, nseed=0, 
    ),

    (2772, 0.1, "bias_y"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        calpha=2, bound=0.1, nseed=9, 
    ),
    (2772, 0.5, "bias_y"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        calpha=2, bound=0.1, nseed=9, 
    ),
    (10296, 0.5, "bias_y"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        calpha=2, bound=0.1, nseed=9, 
    ),
}
