import matplotlib.pyplot as plt
from scripts.plot.plot_core import procurement_load_three_methods, procurement_plot_one

rates = [0.1, 0.4]

fig, axes = plt.subplots(2, len(rates), figsize=(10, 6), sharex=True)
plt.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom = 0.08, wspace=0.12, hspace=0.1)

panel_labels = [(r"(a) $\eta=0.1$", r"(c) $\eta=0.1$"), (r"(b) $\eta=0.4$", r"(d) $\eta=0.4$")]

for i, rate in enumerate(rates):
    df1, df2, df3 = procurement_load_three_methods(rate)

    procurement_plot_one(
        axes[0, i],
        axes[1, i],
        [df1, df2, df3],
        show_ylabel=(i == 0),
        panel_labels=panel_labels[i]
    )
axes[0, 0].set_ylim(-0.2, 17)
axes[1, 0].set_ylim(0.0, 0.12)
axes[0, 1].set_ylim(-0.2, 17)
axes[1, 1].set_ylim(0.0, 0.12)

# 軸ラベル整理
for ax in axes[1]:
    ax.set_xlabel(r"time $t$")

axes[0, 0].legend()

plt.savefig('outputs/power_opt/figures/pdf/deviation_CV_multi.pdf')

plt.show()

