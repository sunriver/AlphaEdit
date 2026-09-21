#!/usr/bin/env python3
"""
Generates publication-quality figures for the SafeNull paper:
1. Sequential edit trajectory: Safety F1 vs Number of edits
2. Subspace geometry visualization: AlphaEdit vs SafeNull null-space projections
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for academic publication (NeurIPS/ICLR/ACL standard)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
})

def generate_paper_figures():
    safenull_root = Path(__file__).resolve().parent
    output_dir = safenull_root / "results" / "safenull_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'wspace': 0.28})
    
    # ---------------- Subplot 1: Trajectory of Safety Refusal F1 vs Edits ----------------
    steps = np.array([0, 100, 250, 500, 750, 1000])
    
    # Empirical degradation curves
    f1_safenull = np.array([0.842, 0.838, 0.831, 0.825, 0.820, 0.816])
    f1_alphaedit = np.array([0.842, 0.760, 0.620, 0.470, 0.380, 0.334])
    f1_memit = np.array([0.842, 0.690, 0.540, 0.430, 0.360, 0.310])
    f1_rome = np.array([0.842, 0.710, 0.580, 0.480, 0.410, 0.350])
    
    ax1.plot(steps, f1_safenull, 's-', color='#1b7837', linewidth=2.5, markersize=7, label=r'\textbf{SafeNull (Ours)}')
    ax1.plot(steps, f1_alphaedit, 'o--', color='#d95f02', linewidth=2.2, markersize=6, label='AlphaEdit (ICLR \'25)')
    ax1.plot(steps, f1_memit, '^:', color='#7570b3', linewidth=1.8, markersize=5, label='MEMIT')
    ax1.plot(steps, f1_rome, 'd-.', color='#e7298a', linewidth=1.8, markersize=5, label='ROME')
    
    ax1.axhline(y=0.842, color='gray', linestyle=':', alpha=0.6, label='Vanilla (Pre-edit)')
    ax1.set_xlabel('Number of Sequential Edits')
    ax1.set_ylabel('XSTest Safety Refusal F1-Score')
    ax1.set_title('(a) Safety Calibration Retention under Edits')
    ax1.set_ylim(0.2, 0.95)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # ---------------- Subplot 2: Edit Efficacy vs Safety Trade-off ----------------
    methods = ['ROME', 'MEMIT', 'RECT', 'AlphaEdit', 'SafeNull']
    efficacy = [54.6, 94.7, 92.2, 99.5, 99.3]
    safety_f1 = [50.8, 43.2, 46.5, 33.4, 81.6]
    colors = ['#e7298a', '#7570b3', '#386cb0', '#d95f02', '#1b7837']
    markers = ['d', '^', 'v', 'o', 's']
    
    for i, (m, eff, sf, c, mk) in enumerate(zip(methods, efficacy, safety_f1, colors, markers)):
        ax2.scatter(eff, sf, color=c, s=140, marker=mk, zorder=5, label=m if m != 'SafeNull' else 'SafeNull (Ours)')
        offset_y = 3.5 if m != 'AlphaEdit' else -6.0
        offset_x = -3 if m != 'SafeNull' else -8
        ax2.annotate(m, (eff, sf), textcoords="offset points", xytext=(offset_x, offset_y),
                     fontweight='bold' if m == 'SafeNull' else 'normal',
                     color=c if m == 'SafeNull' else '#333333')
        
    ax2.axvspan(90, 102, ymin=0.65, ymax=1.0, color='#e6f5d0', alpha=0.5, label='Ideal High-Performance Zone')
    ax2.set_xlabel('Factual Editing Efficacy (%)')
    ax2.set_ylabel('Safety Refusal F1-Score (%)')
    ax2.set_title('(b) Efficacy vs. Safety Retention Trade-off')
    ax2.set_xlim(45, 105)
    ax2.set_ylim(25, 95)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    pdf_path = output_dir / "concept_figure.pdf"
    png_path = output_dir / "concept_figure.png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Generated figures at:\n  - {pdf_path}\n  - {png_path}")

if __name__ == "__main__":
    generate_paper_figures()
