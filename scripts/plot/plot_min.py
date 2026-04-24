from scripts.plot.plot_core import plot_energy

PLOT_CONFIG = {
    "aggregation": "min",
}

plot_energy(
    energy_nb,
    energy_wb,
    target_betas,
    aggregation=PLOT_CONFIG["aggregation"],
    save_path=SAVE_DIR / "min_energy.png"
)
