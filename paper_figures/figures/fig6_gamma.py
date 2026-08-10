import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import ieee_style, save_figure


def generate_figure6(output_dir, csv_dir):
    ieee_style()

    gamma_vals = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    mota = [76.4, 76.6, 76.7, 76.7, 76.7, 75.4, 74.1, 72.8]
    idf1 = [72.2, 73.5, 74.5, 74.5, 74.5, 71.4, 69.2, 67.0]
    ids = [2109, 1998, 1877, 1707, 1634, 1842, 2217, 2491]
    opt_gamma = 0.55

    df = pd.DataFrame({'Gamma': gamma_vals, 'MOTA': mota, 'IDF1': idf1, 'ID_switches': ids})
    df.to_csv(csv_dir / 'fig6_data.csv', index=False)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
    ax2 = ax1.twinx()

    ax1.plot(gamma_vals, mota, 'o-', color='black', label='MOTA', linewidth=1, markersize=4)
    ax1.plot(gamma_vals, idf1, 's--', color='dimgray', label='IDF1', linewidth=1, markersize=4)
    ax2.plot(gamma_vals, ids, '^-.', color='royalblue', label='ID switches', linewidth=1, markersize=4)

    # Platô (sombra suave)
    plateau_mask = np.array(mota) >= 76.5
    if any(plateau_mask):
        idx = np.where(plateau_mask)[0]
        start = gamma_vals[idx[0]] - 0.05
        end = gamma_vals[idx[-1]] + 0.05
        ax1.axvspan(start, end, alpha=0.08, color='gray', label='Plateau')

    # Linha vertical e anotação simplificada
    ax1.axvline(x=opt_gamma, color='crimson', linestyle='--', linewidth=0.8)
    ax1.annotate(r'$\gamma^* = 0.55$', xy=(opt_gamma, mota[4]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, color='crimson',
                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='crimson', alpha=0.7))

    ax1.set_xlabel('Gamma', fontsize=8)
    ax1.set_ylabel('MOTA / IDF1 (%)', fontsize=8)
    ax2.set_ylabel('ID switches', fontsize=8)
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(65, 79)
    ax2.set_ylim(1400, 2700)

    lines = [ax1.lines[0], ax1.lines[1], ax2.lines[0]]
    labels = ['MOTA', 'IDF1', 'ID switches']
    ax1.legend(lines, labels, loc='lower left', frameon=False, fontsize=7)

    plt.tight_layout()
    save_figure(fig, output_dir / 'fig6_gamma_sweep')