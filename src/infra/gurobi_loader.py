from pathlib import Path
import pandas as pd

def load_gurobi_result(nT, L, rate, iseed, mode):
    filename = f"results_nT{nT}_L{L}_MNone_rate{rate}_iseed{iseed}_{mode}.csv"
    path = Path("~/git/gurobi_energy_mathopt/output").expanduser() / filename

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    return pd.read_csv(path)
