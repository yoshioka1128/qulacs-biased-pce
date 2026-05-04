# src/analysis/parser.py

import os


def parse_result_filename(best_file: str):
    """
    result filename から metadata を取り出す

    例:
        results_backprop_alphasc2.5_beta0.1_init4.json
        results_backprop_bias_x_alphasc2.5_beta0.1_init4.json
    """

    metadata = os.path.basename(best_file)

    # .json を除去
    metadata = metadata.replace(".json", "")

    # -----------------------------------------
    # bias_mode / backprop 判定
    # -----------------------------------------
    backprop = "True" if "_backprop" in metadata else "False"

    if "_bias_x_" in metadata:
        bias_mode = "bias_x"
    elif "_bias_y_" in metadata:
        bias_mode = "bias_y"
    else:
        bias_mode = "nobias"

    # -----------------------------------------
    # alphasc / beta / init
    # -----------------------------------------
    try:
        alphasc = float(
            metadata.split("_alphasc")[1].split("_beta")[0]
        )

        beta = float(
            metadata.split("_beta")[1].split("_init")[0]
        )

        init = int(
            metadata.split("_init")[1]
        )

    except (IndexError, ValueError) as e:
        raise ValueError(
            f"failed to parse filename: {best_file}"
        ) from e

    return {
        "bias_mode": bias_mode,
        "backprop": backprop,
        "alphasc": alphasc,
        "beta": beta,
        "init": init,
    }
