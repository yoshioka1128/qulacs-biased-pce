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
    depth: int
    strbp: str
    calpha: float
    bound: float
    nseed: int
    subcounts: int

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
        depth=5, strbp="_backprop",
        calpha=80, bound=1.0, nseed=24, subcounts=4001
    ),

    (60, 0.1, "nobias"): NodeConfig(
        alphasc=6.0, beta=0.1, iinit=4, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=39, subcounts=4001
    ),

    (210, 0.1, "nobias"): NodeConfig(
        alphasc=1.5, beta=0.0, iinit=0, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=0, subcounts=4001
    ),

    (756, 0.1, "nobias"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=4, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=7, subcounts=4001
    ),

    (2772, 0.1, "nobias"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=9, subcounts=4001
    ),

    (18, 0.1, "bias_x"): NodeConfig(
        alphasc=2.5, beta=0.2, iinit=2,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        depth=5, strbp="_backprop",
        calpha=80, bound=1.0, nseed=24, subcounts=4001
    ),

    (60, 0.1, "bias_x"): NodeConfig(
        alphasc=1.0, beta=0.1, iinit=3, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=39, subcounts=4001
    ),

    (210, 0.1, "bias_x"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=3, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=0, subcounts=4001
    ),

    (756, 0.1, "bias_x"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=3, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=7, subcounts=4001
    ),

    (2772, 0.1, "bias_x"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=9, subcounts=4001
    ),

    (18, 0.1, "bias_y"): NodeConfig(
        alphasc=2.5, beta=0.1, iinit=1,
        n_qubits=4, k=2, ninit=5, imax0=1,
        chbetaiinit=[2, 3, 3, 4],
        betas=[-0.1, 0.0, 0.1, 0.2],
        depth=5, strbp="_backprop",
        calpha=80, bound=1.0, nseed=24, subcounts=4001
    ),

    (60, 0.1, "bias_y"): NodeConfig(
        alphasc=1.5, beta=0.0, iinit=4, 
        n_qubits=6, k=3, ninit=5, imax0=2,
        chbetaiinit=[1, 0, 3],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=39, subcounts=4001
    ),

    (210, 0.1, "bias_y"): NodeConfig(
        alphasc=0.5, beta=-0.1, iinit=3, 
        n_qubits=8, k=4, ninit=5, imax0=2,
        chbetaiinit=[0, 4, 2],
        depth=5, strbp="_backprop",
        calpha=6, bound=0.1, nseed=0, subcounts=4001
    ),

    (756, 0.1, "bias_y"): NodeConfig(
        alphasc=0.5, beta=0.0, iinit=3, 
        n_qubits=10, k=5, ninit=5, imax0=3,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=7, subcounts=4001
    ),

    (2772, 0.1, "bias_y"): NodeConfig(
        alphasc=0.1, beta=0.0, iinit=1,
        n_qubits=12, k=6, ninit=5, imax0=5,
        depth=5, strbp="_backprop",
        calpha=2, bound=0.1, nseed=9, subcounts=4001
    ),
}
