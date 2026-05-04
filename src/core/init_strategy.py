import numpy as np

def sample_init(readmode, bias_mode, rng, n_params, init_para=None):
    if readmode:
        if init_para is None: raise ValueError("readmode=True but init_para is None")
        return init_para
    else:
        theta0 = rng.uniform(0, 2*np.pi, size=n_params)
        if bias_mode != "nobias":
            init_para = np.concatenate([theta0, [0.0]])
        else:
            init_para = theta0
    return init_para
