import numpy as np

def generate_spin(spin_mode, n, seed=None):
    if spin_mode == "allzero":
        return np.ones(n, dtype=int)
    elif spin_mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=n)
    else:
        raise ValueError(f"Unknown spin_mode: {spin_mode}")
