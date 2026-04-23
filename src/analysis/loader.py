# loader.y
import re
import json
from pathlib import Path
from collections import defaultdict

pattern = re.compile(r"alphasc([-\d\.]+)_beta([-\d\.]+)")

def load_data(data_dir: Path, use_bias: bool):
    energy_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    loss_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file in data_dir.glob("results_backprop*.json"):
        fname = file.name

        # --- biasフィルタ ---
        if use_bias:
            if "results_backprop_bias_" not in fname:
                continue
        else:
            if "results_backprop_bias_" in fname:
                continue

        # --- reg_type判定 ---
        if use_bias:
            if "_reg_typex" in fname:
                reg_type = "x"
            elif "_reg_typey" in fname:
                reg_type = "y"
            else:
                reg_type = "unknown"
        else:
            reg_type = "no_reg"

        match = pattern.search(fname)
        if not match:
            continue

        alphasc = float(match.group(1))
        beta = float(match.group(2))

        with open(file, "r") as f:
            js = json.load(f)

        energy = js["Calculated Minimum Energy [norm, row]"][0]
        loss = js["Corresponding loss function"]

        energy_data[beta][alphasc][reg_type].append(energy)
        loss_data[beta][alphasc][reg_type].append(loss)

    return energy_data, loss_data
