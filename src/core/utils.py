from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob, os, json, re
from pathlib import Path
from qulacs import QuantumState, Observable, PauliOperator
from gurobi_energy_mathopt import utils as gurobi_utils

def convert_seconds_to_hms(seconds):
    # Convert seconds to hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return hours, minutes, seconds

# plot optimization process
def get_binary_solution(n_nodes, para, ansatz, hamiltonian, n_qubits, USE_BIAS=False, alpha=1.0):

    # Set the ansatz parameters and update the quantum state
    if USE_BIAS:
        theta = para[:-1]
        bias = para[-1]
    else:
        theta = para
        bias = None
    ansatz.set_parameter(theta)
    state = QuantumState(n_qubits)
    ansatz.update_quantum_state(state)

    # Compute the expectation values for each term in the Hamiltonian
    exp_value = np.zeros(n_nodes)
    for i in range(hamiltonian.get_term_count()):
        h_term = hamiltonian.get_term(i)
        exp_value[i] = h_term.get_expectation_value(state).real

    # Convert the expectation values to binary values (using sign function)
    z = alpha * exp_value + (bias if bias is not None else 0.0)
    binary_solution = np.sign(z)

    return binary_solution

def number_to_spin(number, n_spin):
    # Convert a number to a binary spin representation (1 for '0', -1 for '1')
    binary_str = format(number, "b").zfill(n_spin)
    spin_list = np.array([1 if bit == '0' else -1 for bit in binary_str])

    # Flip the array so that the least significant bit is first
    return np.flipud(spin_list)


def spin_to_number(spin_array):
    # Reverse the spin array
    reversed_spin_array = spin_array[::-1]

    # Convert spins to bits (1 -> 0, -1 -> 1)
    bit_array = (reversed_spin_array == -1).astype(int)

    # Convert the bit array to a decimal number
    decimal_value = int("".join(bit_array.astype(str)), 2)

#    print("Bit array:", bit_array)
#    print("Decimal value:", decimal_value)
    return decimal_value

def prepare_int_from_d(consumer_list, m, it, nT, rate, iseed=42):
    base_dir = Path("outputs/power_opt/numpy_param")

    # ファイルパス
    dJ_file = base_dir / f"dJ_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy"
    dhex_file = base_dir / f"dhex_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy"
    shift_file = base_dir / f"shift_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy"
    frob_file = base_dir / f"frob_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy"

    # --- 既存ファイルチェック ---
    if all(p.exists() for p in [dJ_file, dhex_file, shift_file, frob_file]):
        dJ = np.load(dJ_file)
        dhex = np.load(dhex_file)
        shift = np.load(shift_file).item()
        frob_norm = np.load(frob_file).item()

        return frob_norm, shift, dJ, dhex

    # --- 計算モード ---
    large_str = "_large" if m == 10296 else ""

    sigma, Pt_ex, proc = gurobi_utils.load_covariance_and_expectation(
        True, consumer_list, it, nT, rate, large_str
    )

    frob_norm = gurobi_utils.compute_qubo_frobenius_norm(
        sigma, Pt_ex, proc, nT
    )

    dJ, dhex, shift = qubo2ising_correct(sigma, Pt_ex, proc)

    # 正規化
    dJ = dJ / frob_norm
    dhex = dhex / frob_norm

    # --- 保存 ---
    base_dir.mkdir(parents=True, exist_ok=True)

    np.save(dJ_file, dJ)
    np.save(dhex_file, dhex)
    np.save(shift_file, shift)
    np.save(frob_file, frob_norm)

    return frob_norm, shift, dJ, dhex

#def prepare_int_from_d(consumer_list, m, it, nT, rate, iseed=42):
#    # 保存ファイル名（ディレクトリ d 内に作る）
#    dJ_file = os.path.join(f"outputs/power_opt/numpy_param/dJ_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy")
#    dhex_file = os.path.join(f"outputs/power_opt/numpy_param/dhex_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy")
#    shift_file = os.path.join(f"outputs/power_opt/numpy_param/shift_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy")
#    frob_file = os.path.join(f"outputs/power_opt/numpy_param/frob_it{it}_nT{nT}_rate{rate}_{m}nodes_iseed{iseed}.npy")
#
#    # --- 既存ファイルがあればロード ---
#    if all(os.path.isfile(f) for f in [dJ_file, dhex_file, shift_file, frob_file]):
#        dJ = np.load(dJ_file)
#        dhex = np.load(dhex_file)
#        shift = np.load(shift_file).item()
#        frob_norm = np.load(frob_file).item()
#    else:
#        # 計算する
#        if m == 10296: large_str = "_large"
#        else: large_str = ""
#        sigma, Pt_ex, proc = gurobi_utils.load_covariance_and_expectation(True, consumer_list, it, nT, rate, large_str)
#        frob_norm = gurobi_utils.compute_qubo_frobenius_norm(sigma, Pt_ex, proc, nT)
#        dJ, dhex, shift = qubo2ising_correct(sigma, Pt_ex, proc)
#
#        # 正規化
#        dJ = dJ / frob_norm
#        dhex = dhex / frob_norm
#
#        # 保存（npy形式で高速 & 精度保持）
#        Path(dJ_file).parent.mkdir(parents=True, exist_ok=True)
#        np.save(dJ_file, dJ)
#        np.save(dhex_file, dhex)
#        np.save(shift_file, shift)
#        np.save(frob_file, frob_norm)
#
#    # グラフ構築
#    return frob_norm, shift, dJ, dhex

def iterative_greedy_ising(J, h, z0=None, prev_E=np.inf, max_iter=1000, tol=1e-12):
    """
    Apply greedy_ising repeatedly until energy stops decreasing.

    Parameters
    ----------
    J : ndarray
        Interaction matrix
    h : ndarray
        Local field
    z0 : list or ndarray
        Initial spin configuration
    max_iter : int
        Maximum number of iterations
    tol : float
        Energy tolerance for convergence

    Returns
    -------
    z : list
        Final spin configuration
    E : float
        Final energy
    n_iter : int
        Number of iterations
    """

    z = z0
#    prev_E = np.inf

    for it in range(max_iter):

        z, E = greedy_ising(J, h, z)

        if abs(prev_E - E) < tol:
            break

        prev_E = E
        print(it+2, E)

    return z, E, it + 1

def greedy_ising(J, h, z0=None):
    """
    Apply a greedy spin-flip algorithm to the Ising model (J, h).
    Spins are treated directly as z ∈ {-1, +1}.

    Parameters
    ----------
    J : ndarray (N, N)
        Symmetric interaction matrix (diagonal elements assumed to be 0)
    h : ndarray (N,)
        Local magnetic field
    z0 : ndarray (N,), optional
        Initial spin configuration (elements can be ±1 or 0).
        If 0, the value is first greedily assigned to -1 or +1.
        If None, all spins are initialized to +1.

    Returns
    -------
    z : ndarray (N,)
        Final spin configuration (±1)
    E : float
        Final energy
    """
    N = len(h)

    # Initialize spins
    if z0 is None:
        z = np.ones(N, dtype=int)
    else:
        z = np.array(z0, dtype=int).copy()
        if not np.all(np.isin(z, [-1, 0, 1])):
            raise ValueError("Elements of z0 must be -1, 0, or +1")

    # --- Step 1: Assign spins where z=0 greedily ---
    for i in range(N):
        if z[i] == 0:
            # Compute effective local field using currently assigned spins
            effective_field = h[i] + np.dot(J[i], z * (z != 0))  # ignore unassigned spins
            if effective_field > 0:
                z[i] = -1
            elif effective_field < 0:
                z[i] = +1
            else:
                z[i] = np.random.choice([-1, 1])  # tie-break

    # --- Step 2: Apply standard greedy spin-flip ---
    chosen = np.zeros(N, dtype=bool)
    local_field = h + J @ z

    while True:
        delta = np.full(N, np.inf)
        for j in range(N):
            if not chosen[j]:
                # ΔE_j = -2 z_j (h_j + sum_i J_ij z_i)
                delta[j] = -2 * z[j] * local_field[j]

        j_best = np.argmin(delta)
        if delta[j_best] >= 0:
            break

        # スピン反転
        z[j_best] *= -1
        chosen[j_best] = True

        # local_field を更新
        local_field += 2 * J[:, j_best] * z[j_best]

    # エネルギー計算
    E = float(z @ np.tril(J, -1) @ z + h @ z)

    return z.tolist(), E


def greedy_best(J, h, z0=None, max_iter=10_000):
    """
    Greedy single-spin flip local search (steepest descent)

    Parameters
    ----------
    z : np.ndarray
        Initial spin configuration (±1)
    J : np.ndarray
        Coupling matrix (NxN)
    h : np.ndarray
        Local field (N)
    max_iter : int
        Safety cap to avoid infinite loops

    Returns
    -------
    z : list
        Optimized spin configuration
    E : float
        Final energy
    """

#    z = z.copy()
    N = len(h)

    # Initialize spins
    if z0 is None:
        z = np.ones(N, dtype=int)
    else:
        z = np.array(z0, dtype=int).copy()
        if not np.all(np.isin(z, [-1, 0, 1])):
            raise ValueError("Elements of z0 must be -1, 0, or +1")

    # 初期局所場
    local_field = h + J @ z

    # =========================================================
    # Step 1: 0スピンを best-first で埋める
    # =========================================================
    zero_indices = list(np.where(z == 0)[0])

    while len(zero_indices) > 0:
        best_i = None
        best_delta = np.inf
        best_spin = None

        for i in zero_indices:
            lf = local_field[i]

            # +1 / -1 のエネルギー変化
            delta_plus  = - lf
            delta_minus = + lf

            if delta_plus < delta_minus:
                delta_i = delta_plus
                spin_i = +1
            else:
                delta_i = delta_minus
                spin_i = -1

            # best-first選択
            if delta_i < best_delta:
                best_delta = delta_i
                best_i = i
                best_spin = spin_i

        # 決定
        z[best_i] = best_spin

        # 局所場更新（0 → ±1）
        local_field += J[:, best_i] * z[best_i]

        # 未確定集合から削除
        zero_indices.remove(best_i)

    # =========================================================
    # Step 2: greedy local search（制限なし版）
    # =========================================================
    for _ in range(max_iter):

        # ΔE の計算
        delta = -2 * z * local_field

        # 最良スピン
        j_best = np.argmin(delta)

        # 改善がなければ終了
        if delta[j_best] >= 0:
            break

        # スピン反転
        z[j_best] *= -1

        # 局所場の更新（高速）
        local_field += 2 * J[:, j_best] * z[j_best]

    # エネルギー計算
    E = float(z @ np.tril(J, -1) @ z + h @ z)

    return z.tolist(), E


def greedy_qubo_from_ising(J, h):
    """
    J,h (Ising) を受け取り、
    z = 1 - 2x 変換で得た QUBO に
    0/1 貪欲法を適用
    """
    Q, c, c0 = ising_to_qubo_z1minus2x(J, h)
    N = len(h)
    x = np.zeros(N, dtype=int)
    chosen = np.zeros(N, dtype=bool)

    # Sとの相互作用累積を保持してO(kN)化
    Qx = np.zeros(N)

    while True:
        delta = np.full(N, np.inf)
        for j in range(N):
            if not chosen[j]:
                # Δ_j = c_j + Q_{jj} + 2 * sum_{i∈S} Q_{ij}
                delta[j] = c[j] + Q[j, j] + Qx[j]

        j_best = np.argmin(delta)
        if delta[j_best] >= 0:
            break

        x[j_best] = 1
        chosen[j_best] = True
        Qx += Q[:, j_best]

    cost = float(x @ np.tril(Q, -1) @ x + c @ x + c0)
    z = 1 - 2 * x
    E_ising = float(z @ np.tril(J, -1) @ z + h @ z)
    
    return z.tolist(), cost


def local_search_ising_deltaE(J, h, x):
    """
    J: 相互作用行列 (2D numpy array)
    h: ローカルフィールド (1D numpy array)
    x: スピン配置 (±1 の numpy array)
    """
    N = len(x)
    # 初期エネルギー
    E = np.sum(np.tril(J, -1) * np.outer(x, x)) + np.dot(h, x)
    improved = True
    while improved:
        improved = False
        for k in range(N):
            # ΔEを計算
            deltaE = - 2 * x[k] * (h[k] + np.dot(J[k], x))
            if deltaE < 0:
                # 反転
                x[k] *= -1
                E += deltaE
                improved = True
                break
    return x, E


def local_search_ising_deltaE_onepass(J, h, x0=None):
    """
    One-pass local search for Ising model using deltaE criterion.
    If x0 contains 0 entries (unassigned), they are first greedily assigned
    to -1 or +1 based on the local effective field, then a single pass of
    deltaE-based flips is performed.

    Parameters
    ----------
    J : 2D numpy array
        Interaction matrix (shape (N, N))
    h : 1D numpy array
        Local fields (shape (N,))
    x0 : array-like of length N, optional
        Initial spin configuration. Elements may be -1, 0, or +1.
        If None, all spins are initialized to +1.

    Returns
    -------
    x : numpy.ndarray
        Final spin configuration (elements are -1 or +1)
    E : float
        Energy under the convention E = sum_{i<j} J_ij x_i x_j + sum_i h_i x_i
    """
    N = len(h)

    # initialize x (allow -1, 0, +1 in x0)
    if x0 is None:
        x = np.ones(N, dtype=int)
    else:
        x = np.array(x0, dtype=int).copy()
        if not np.all(np.isin(x, [-1, 0, 1])):
            raise ValueError("x0 elements must be -1, 0, or +1")

    # --- Step 1: greedily assign any unassigned spins (x == 0) ---
    # Use current spins (with 0 meaning 'unassigned' so their contribution is zero)
    for i in range(N):
        if x[i] == 0:
            # effective local field (ignores other unassigned spins because they are zero)
            effective_field = h[i] + np.dot(J[i], x)
            if effective_field > 0:
                x[i] = -1
            else:
                # effective_field < = 0 -> choose +1 (deterministic tie-break)
                x[i] = +1

    # --- compute initial energy (same convention as original function) ---
    E = np.sum(np.tril(J, -1) * np.outer(x, x)) + np.dot(h, x)

    # --- Step 2: one-pass local search using deltaE ---
    for k in range(N):
        deltaE = -2 * x[k] * (h[k] + np.dot(J[k], x))
        if deltaE < 0:
            x[k] *= -1
            E += deltaE

    return x.tolist(), E

#def local_search_ising_deltaE_onepass(J, h, x0=None):
#    """
#    J: 相互作用行列 (2D numpy array)
#    h: ローカルフィールド (1D numpy array)
#    x: スピン配置 (±1 の numpy array)
#    """
#    N = len(h)
#
#    # 初期状態
#    if x0 is None:
#        x = [1 for _ in range(N)]
#    else:
#        x = x0.copy()
#        if not np.all(np.isin(x, [-1, 1])):
#            raise ValueError("z0 の要素は ±1 である必要があります")
#
#    # 初期エネルギー
#    E = np.sum(np.tril(J, -1) * np.outer(x, x)) + np.dot(h, x)
#
#    # --- 全頂点に対して1ラウンド探索 ---
#    for k in range(N):
#        deltaE = -2 * x[k] * (h[k] + np.dot(J[k], x))
#        if deltaE < 0:
#            # 反転して更新
#            x[k] *= -1
#            E += deltaE
#
#    return x, E

def ising_to_qubo_z1minus2x(J, h):
    """
    Ising (±1) -> QUBO (0/1)
    変換 z = 1 - 2x
    """
    Q = 4 * J
    c = -2 * np.sum(J, axis=1) - 2 * h
    c0 = float(np.sum(np.tril(J, -1)) + np.sum(h))  # 1^T J 1 + h^T 1
    return Q, c, c0

def qubo2ising_correct(sigma, P, proc):
    L, nT = P.shape
    # A, b, c を明示的に作る
    A = sigma + (P @ P.T) / nT
    b = -(2.0 / nT) * (P @ proc)          # shape (L,)
    c = np.mean(proc**2)                   # = (1/nT) sum proc_t^2

    one = np.ones(L)

    # エッジ重み（NetworkXの無向エッジ1回分の重み）
    # 完全グラフを想定: i<j のみに設定
    W = np.zeros((L, L))
    iu = np.tril_indices(L, k=-1)
    W[iu] = A[iu] / 2.0

    # 局所場
    h = -0.5 * (A @ one + b)

    # 定数項（shift）
    shift = c + 0.25 * (one @ A @ one) + 0.5 * (one @ b) + 0.25 * np.trace(A)

    return W, h, shift


def find_best_in_group(file_list):
    best_file = None
    best_energy = float("inf")
    best_data = None
    best_params = None

    for f in file_list:
        with open(f, "r") as fp:
            data = json.load(fp)
        iterations = data["Iterations"]
        energy_value = data["Calculated Minimum Energy [norm, row]"][0]
        if energy_value < best_energy:
            best_energy = energy_value
            best_file = f

    if best_file is None:
        return None

    return best_file

def parse_filename(d: str):
    """
    文字列 d から time, nT, rate, nodes を抽出する
    - time, nT, nodes は int
    - rate は float
    """
    pattern = r"time(\d+)_nT(\d+)_rate([\d.]+)_(\d+)nodes_(\d+)qubits_(\d+)body_ninit(\d+)"
    match = re.search(pattern, d)
    if not match:
        raise ValueError(f"Could not parse: {d}")
    
    time = int(match.group(1))
    nT = int(match.group(2))
    rate = float(match.group(3))
    nodes = int(match.group(4))
    qubits = int(match.group(5))
    body = int(match.group(6))
    ninit = int(match.group(7))
    
    return time, nT, rate, nodes, qubits, body, ninit

def build_record_from_best_file(best_file, nodes, qubits, body):

    param_re = re.compile(r"results(_backprop)?_alpha([0-9.]+)_beta([0-9.]+)_init(\d+)")
    m = param_re.search(best_file)
    if m is None:
        return None
    bp = "True" if m.group(1) is not None else "False"
    alpha = float(m.group(2))
    beta = float(m.group(3))
    init = int(m.group(4))

    with open(best_file, "r") as f:
        data = json.load(f)

    return {
        "nodes": nodes,
        "qubits": qubits,
        "body": body,
        "alpha": alpha,
        "alphasc": alpha / (qubits ** np.floor(body / 2)),
        "beta": beta,
        "init": init,
        "backprop": bp,
        "frob_norm": data["Cmin, Cmax, frob_norm, shift"][2],
        "energy": data["Calculated Minimum Energy [norm, row]"][0],
        "nparams": data["Number of Parameters"],
        "niter": data["Iterations"],
        "elapsed": data["Elapsed Time [seconds]"],
        "best_file": best_file,
        "solution": data["Solution for Minimum Energy"],
        "normalize": data["Cmin, Cmax, frob_norm, shift"]
    }


def make_consumer_color_dict(
    consumer_list_file,
    cmap_names=("tab20", "tab20b", "tab20c"),
):
    consumer_df = pd.read_csv(consumer_list_file)
    consumer_list_all = consumer_df["Consumer"].tolist()
    nodes = len(consumer_list_all)

    # --- 60色を作る ---
    color_list = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        color_list.extend(cmap.colors)  # 各20色

    color_list = np.array(color_list)  # shape (60, 4)
    
    colors = np.array([color_list[i % len(color_list)] for i in range(nodes)])

    return dict(zip(consumer_list_all, colors)), consumer_list_all

def make_consumer_color_dict20(
    consumer_list_file,
    cmap_name="tab20"
):
    """
    Consumer ID → color の辞書を作成する

    Parameters
    ----------
    consumer_list_file : str
        Consumer一覧CSV（Consumer列を含む）
    cmap_name : str
        matplotlibのカラーマップ名（default: tab20）

    Returns
    -------
    dict
        {consumer_id: RGBA color}
    """
    consumer_df = pd.read_csv(consumer_list_file)
    consumer_list_all = consumer_df["Consumer"].tolist()
    nodes = len(consumer_list_all)

    cmap = plt.get_cmap(cmap_name)
    color_list = cmap(np.linspace(0, 1, cmap.N))

    colors = [color_list[i % len(color_list)] for i in range(nodes)]

    return dict(zip(consumer_list_all, colors)), consumer_list_all

def make_consumer_color_dict_continuous(
    consumer_list_file,
    cmap_name="hsv",   # ← 連続カラーマップ
):
    consumer_df = pd.read_csv(consumer_list_file)
    consumer_list_all = consumer_df["Consumer"].tolist()
    nodes = len(consumer_list_all)

    cmap = plt.get_cmap(cmap_name)

    # nodes 個を [0,1] で等間隔にサンプリング
    colors = cmap(np.linspace(0, 1, nodes))

    return dict(zip(consumer_list_all, colors)), consumer_list_all

def distinct_colors_hsl(n, s=0.65, l=0.55):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [mcolors.hsv_to_rgb((h, s, l)) for h in hues]
    return colors

def one_sample(nodes, dJ_sym, dhex, seed):
    np.random.seed(seed)
    x_greedy2 = np.random.choice([-1, 1], size=nodes)
    x_greedy2, cost_greedy2 = greedy_ising(dJ_sym, dhex, x_greedy2)
    return cost_greedy2, x_greedy2

def get_obj_values_by_hour(csv_file: str, hour_value: int):
    df = pd.read_csv(csv_file)

    # 指定hourでフィルタ
    df_hour = df[df["hour"] == hour_value]

    if df_hour.empty:
        raise ValueError(f"hour={hour_value} がCSV内に存在しません")

    # 最初の1行を取得
    row = df_hour.iloc[0]

    return (
        row["obj_val_min"],
        row["obj_val_max"],
        row["frobenius_norm"]
    )
