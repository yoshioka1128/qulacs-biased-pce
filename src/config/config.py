# config.py
from dataclasses import dataclass

@dataclass
class Config:
    use_new: bool = True
    learn: bool = False
    nprob: int = 1
    ninit: int = 5
    iseed: int = 42
    method: str = "BFGS"
    verbose: int = 0
    maxiter: int = 10000
    readmode: bool = False
    backprop: bool = True
    bias: bool = False
