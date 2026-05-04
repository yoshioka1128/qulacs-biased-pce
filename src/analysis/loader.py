import re
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from src.config.problem_config import PROBLEM_CONFIG
from src.config.full_config import FullConfig

pattern = re.compile(
    r"alphasc(?P<alpha>[-\d\.]+)_beta(?P<beta>[-\d\.]+)_init(?P<init>\d+)"
)

def get_result_file_from_node_config(
    cfg: FullConfig,
    nodes: int,
    rate: float,
    model: str,
    mode: str,
    method: str,
    iseed: int,
    type_ansatz: str = "all2all",
    it: int =1,
    nT: int = 24,
    readmode: bool = False,
):

    key = (nodes, rate, model)

    if key not in PROBLEM_CONFIG:
        raise ValueError(f"PROBLEM_CONFIG not found: {key}")

    alphasc = cfg.alphasc
    beta = cfg.beta
    iinit = cfg.iinit
    n_qubits = cfg.n_qubits
    k = cfg.k
    depth = cfg.depth
    ninit = cfg.ninit
    strbp = cfg.strbp

#    suffix = f"{strbp}_{mode}"
    if mode == "nobias":
        suffix = strbp
    elif mode == "bias_x":
        suffix = f"{strbp}_bias_x"
    elif mode == "bias_y":
        suffix = f"{strbp}_bias_y"
    else:
        raise ValueError(f"unknown mode: {mode}")

    if readmode:
        read_dir = (
            f"outputs/power_opt/"
            f"time{it}_nT1_rate{rate}_"
            f"{nodes}nodes_{n_qubits}qubits_{k}body_"
            f"ninit{ninit}_depth{depth}_"
            f"{type_ansatz}_method{method}_"
            f"iseed{iseed}/read"
        )
        read_file = os.path.join(
            read_dir,
            f"results{suffix}_"
            f"alphasc{alphasc}_"
            f"beta{beta}_"
            f"init0.json"
        )
    else:
        read_dir = (
            f"outputs/power_opt/"
            f"time1_nT24_rate{rate}_"
            f"{nodes}nodes_{n_qubits}qubits_{k}body_"
            f"ninit{ninit}_depth{depth}_"
            f"{type_ansatz}_method{method}_"
            f"iseed{iseed}"
        )
        read_file = os.path.join(
            read_dir,
            f"results{suffix}_"
            f"alphasc{alphasc}_"
            f"beta{beta}_"
            f"init{iinit}.json"
        )

    if not os.path.exists(read_file):
        raise FileNotFoundError(
            f"result file not found:\n{read_file}"
        )

    return read_dir, read_file


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

def load_result_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def build_result_record(
    meta: dict,
    data: dict,
    nodes: int,
    qubits: int,
    body: int,
    best_file: str = None,
):
    """
    parse_result_filename() と
    load_result_json() の結果から
    record を作る
    """

    alphasc = meta["alphasc"]
    beta = meta["beta"]
    init = meta["init"]
    mode = meta["mode"]
    backprop = meta["backprop"]

    # legacy互換:
    # alpha を復元
    alpha = alphasc * (qubits ** np.floor(body / 2))

    return {
        "nodes": nodes,
        "qubits": qubits,
        "body": body,
        "mode": mode,
        "alpha": alpha,
        "alphasc": alphasc,
        "beta": beta,
        "init": init,
        "backprop": backprop,
        "frob_norm": data["Cmin, Cmax, frob_norm, shift"][2],
        "cost w/o pp": data["Calculated Minimum Energy [norm, row]"][0],
        "loss": data["Corresponding loss function"],
        "nparams": data["Number of Parameters"],
        "niter": data["Iterations"],
        "elapsed": data["Elapsed Time [seconds]"],
        "best_file": best_file,
        "solution": data["Solution for Minimum Energy"],
        "normalize": data["Cmin, Cmax, frob_norm, shift"],
    }
