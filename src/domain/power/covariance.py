import numpy as np


def load_covariance(hour, str_large=""):
    file = f"param-enemane/param/covariance_matrix_time{hour:02d}_mixup_restricted{str_large}.npz"

    with np.load(file, allow_pickle=True) as data:
        cov_matrix = data["cov"]
        all_names = data["names"]

    return cov_matrix, all_names


def build_index_map(all_names, consumer_list):
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    idx = np.array([name_to_idx[c] for c in consumer_list])
    return idx


def extract_sub_covariance(cov_matrix, idx):
    return cov_matrix[np.ix_(idx, idx)]
