import numpy as np

def generate_spin(mode, n, seed=None):
    if mode == "allzero":
        return np.ones(n, dtype=int)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=n)
    else:
        raise ValueError(f"Unknown mode: {mode}")
