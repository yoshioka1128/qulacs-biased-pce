# aggregator.py
import numpy as np

def aggregate(values, mode="mean"):
    if mode == "mean":
        return np.mean(values)
    elif mode == "min":
        return np.min(values)
    elif mode == "median":
        return np.median(values)
    else:
        raise ValueError(f"Unknown mode: {mode}")
