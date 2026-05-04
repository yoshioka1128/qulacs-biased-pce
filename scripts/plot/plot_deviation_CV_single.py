import matplotlib.pyplot as plt
from scripts.plot.plot_core import procurement_load_three_methods, procurement_plot_one
import pltutils

def main():
    rate = 0.1  # ← 必要なら argparseで外出し可

    df1, df2, df3 = procurement_load_three_methods(rate)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)

    procurement_plot_one(
        axes[0],
        axes[1],
        [df1, df2, df3],
        title=f"rate = {rate}"
    )

    # 軸ラベルなど
    axes[1].set_xlabel("hours")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].grid(True)

    plt.show()


if __name__ == "__main__":
    main()
