import re
import json
from pathlib import Path
from collections import defaultdict

pattern = re.compile(
    r"alphasc(?P<alpha>[-\d\.]+)_beta(?P<beta>[-\d\.]+)_init(?P<init>\d+)"
)

def load_data(data_dir: Path, use_bias: bool):
    energy_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    loss_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for file in data_dir.glob("results_backprop*.json"):
        fname = file.name

        # -------------------
        # bias filter
        # -------------------
        is_bias = "results_backprop_bias_" in fname

        if use_bias and not is_bias:
            continue

        if not use_bias and is_bias:
            continue

        # -------------------
        # reg_type
        # -------------------
        if use_bias:
            if "results_backprop_bias_x_" in fname:
                reg_type = "x"
            elif "results_backprop_bias_y_" in fname:
                reg_type = "y"
            else:
                continue
        else:
            reg_type = "no_reg"

        # -------------------
        # parse filename
        # -------------------
        match = pattern.search(fname)
        if not match:
            continue

        alphasc = float(match.group("alpha"))
        beta = float(match.group("beta"))

        # -------------------
        # load json
        # -------------------
        try:
            with open(file, "r") as f:
                js = json.load(f)

            energy = js["Calculated Minimum Energy [norm, row]"][0]
            loss = js["Corresponding loss function"]

        except Exception as e:
            print(f"ERROR: {file}")
            print(e)
            continue

        energy_data[beta][alphasc][reg_type].append(energy)
        loss_data[beta][alphasc][reg_type].append(loss)

    return energy_data, loss_data
