import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/inference"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import ieee_style, save_figure


def generate_figure5(output_dir, csv_dir):
    ieee_style()

    # Dados originais (inalterados)
    stages_full = ['Baseline', '+ GM-FLIT\n+ NSA', '+ Hard\nMining', '+ SAHI']
    stages = ['Baseline', '+FLIT+NSA', '+Hard Mining', '+SAHI']
    mota = [75.9, 76.7, 78.5, 79.9]
    idf1 = [71.2, 74.5, 77.8, 82.7]
    ids = [2327, 1634, 1089, 494]

    df = pd.DataFrame({'Stage': stages_full, 'MOTA': mota, 'IDF1': idf1, 'ID_switches': ids})
    df.to_csv(csv_dir / 'fig5_data.csv', index=False)

    x = np.arange(len(stages))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
    ax2 = ax1.twinx()

    color_mota = '#2C3E50'
    color_idf1 = '#5DADE2'
    color_ids = '#E67E22'

    # Barras
    bars1 = ax1.bar(x - width/2, mota, width, color=color_mota, label='MOTA')
    bars2 = ax1.bar(x + width/2, idf1, width, color=color_idf1, label='IDF1')

    # Curva ID switches
    line_ids, = ax2.plot(x, ids, 'o-', color=color_ids, linewidth=0.7,
                         markersize=2.5, markerfacecolor='white', markeredgewidth=0.5,
                         label='ID switches', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylabel('MOTA / IDF1 (%)', fontsize=8)
    ax2.set_ylabel('ID switches', fontsize=8)
    ax1.set_ylim(65, 85)
    ax2.set_ylim(400, 2500)

    ax1.grid(False)
    ax2.grid(False)

    # Legenda
    lines = [bars1, bars2, line_ids]
    labels = ['MOTA', 'IDF1', 'ID switches']
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.10),
               ncol=3, frameon=False, fontsize=7)

    # --- Rótulos apenas no primeiro e último estágio ---
    offset_vert = 5
    first_idx = 0
    last_idx = len(stages) - 1

    # MOTA: primeiro e último
    for idx in [first_idx, last_idx]:
        ax1.annotate(f'{mota[idx]:.1f}', xy=(x[idx] - width/2, mota[idx]),
                     xytext=(0, offset_vert), textcoords='offset points',
                     ha='center', fontsize=6, color=color_mota, fontweight='bold')

    # IDF1: primeiro e último
    for idx in [first_idx, last_idx]:
        ax1.annotate(f'{idf1[idx]:.1f}', xy=(x[idx] + width/2, idf1[idx]),
                     xytext=(0, offset_vert), textcoords='offset points',
                     ha='center', fontsize=6, color=color_idf1, fontweight='bold')

    # --- Anotações ID switches (apenas primeiro e último, com caixa branca) ---
    bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=color_ids, alpha=0.7, linewidth=0.5)

    ax2.annotate(f'{ids[first_idx]}', xy=(x[first_idx], ids[first_idx]), xytext=(8, 0),
                 textcoords='offset points', ha='left', va='center',
                 fontsize=6, color=color_ids, fontweight='bold', bbox=bbox_props)

    ax2.annotate(f'{ids[last_idx]}', xy=(x[last_idx], ids[last_idx]), xytext=(8, 0),
                 textcoords='offset points', ha='left', va='center',
                 fontsize=6, color=color_ids, fontweight='bold', bbox=bbox_props)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, output_dir / 'fig5_pipeline_evolution')