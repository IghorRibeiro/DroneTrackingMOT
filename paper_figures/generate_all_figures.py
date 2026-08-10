#!/usr/bin/env python3
"""
Orquestra a geração de todas as figuras do artigo.
"""
import sys
from pathlib import Path

# Adiciona o caminho para importar módulos comuns (se necessário)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/inference"))

from figures.fig4_root_cause import generate_figure4
from figures.fig5_pipeline import generate_figure5
from figures.fig6_gamma import generate_figure6



def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'output'
    csv_dir = base_dir / 'csv'
    output_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

   #print("Generating Figure 4 – Root Cause Analysis...")
   #generate_figure4(output_dir, csv_dir)

    print("Generating Figure 5 – Pipeline Evolution...")
    generate_figure5(output_dir, csv_dir)

    #print("Generating Figure 6 – Gamma Sweep...")
    #generate_figure6(output_dir, csv_dir)

    #print("Generating Figure 7 – Qualitative Example...")
    #generate_figure7(output_dir, csv_dir)

    print(f"\nAll figures saved in {output_dir}")
    print(f"CSV data saved in {csv_dir}")


if __name__ == "__main__":
    main()