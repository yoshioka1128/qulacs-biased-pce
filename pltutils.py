import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["axes.axisbelow"] = True

import matplotlib.pyplot as plt

plt.rcParams.update({
    "xtick.top": True,
    "ytick.right": True,
})

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
    \usepackage{newtxtext}
    \usepackage{newtxmath}
    """,
#    "font.family": "serif",
#    "text.latex.preamble": r"\usepackage{mathptmx}",  # Times Roman を使う
    "font.size": 16,             # 全体フォントサイズ
    "axes.labelsize": 16,        # x/yラベル
    "xtick.labelsize": 14,       # x軸目盛
    "ytick.labelsize": 14,       # y軸目盛
    "legend.fontsize": 14,       # 凡例
    "axes.titlesize": 18,        # タイトル
})

plt.rcParams["font.family"] = "Times New Roman"
