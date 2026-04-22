import numpy as np

def sample_init(readmode, rng, n_params, init_para=None):
    if readmode:
        if init_para is None: raise ValueError("readmode=True but init_para is None")
        return init_para
    return rng.uniform(0, 2*np.pi, size=n_params)
