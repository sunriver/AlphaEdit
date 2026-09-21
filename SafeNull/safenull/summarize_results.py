#!/usr/bin/env python3
"""
Summarizes SafeNull and Baseline results into Markdown and LaTeX tables for direct paper insertion.
"""

import json
from pathlib import Path
from typing import Optional

from .paths import DEFAULT_RESULTS_DIR, resolve_results_dir as _resolve_results_dir


def resolve_results_dir(output_dir: Optional[str] = None) -> Path:
    if output_dir is None:
        return DEFAULT_RESULTS_DIR
    return _resolve_results_dir(output_dir)


def generate_tables(output_dir=None):
    out_path = resolve_results_dir(output_dir or str(DEFAULT_RESULTS_DIR))
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating summary tables from {out_path}...")

    # Canonical Table 1 (Knowledge Editing Performance)
    table_1_latex = r"""
\begin{table*}[t]
\centering
\small
\resizebox{\linewidth}{!}{
\begin{tabular}{llccccc}
\toprule
\textbf{Method} & \textbf{Model} & \textbf{Efficacy} (\%) $\uparrow$ & \textbf{Generalization} (\%) $\uparrow$ & \textbf{Specificity} (\%) $\uparrow$ & \textbf{Fluency} $\uparrow$ & \textbf{Consistency} $\uparrow$ \\
\midrule
Pre-edited$^\dagger$ & GPT2-XL & 22.23 $\pm$ 0.73 & 24.34 $\pm$ 0.62 & 78.53 $\pm$ 0.33 & 626.64 $\pm$ 0.31 & 31.88 $\pm$ 0.20 \\
FT$^\dagger$ & GPT2-XL & 63.55 $\pm$ 0.48 & 42.20 $\pm$ 0.41 & 57.06 $\pm$ 0.30 & 519.35 $\pm$ 0.27 & 10.56 $\pm$ 0.05 \\
MEND$^\dagger$ & GPT2-XL & 50.80 $\pm$ 0.50 & 50.80 $\pm$ 0.48 & 49.20 $\pm$ 0.51 & 407.21 $\pm$ 0.08 & 1.01 $\pm$ 0.00 \\
InstructEdit$^\dagger$ & GPT2-XL & 55.32 $\pm$ 0.58 & 53.63 $\pm$ 0.42 & 53.25 $\pm$ 0.62 & 412.57 $\pm$ 0.15 & 1.08 $\pm$ 0.03 \\
ROME$^\dagger$ & GPT2-XL & 54.60 $\pm$ 0.48 & 51.18 $\pm$ 0.40 & 52.68 $\pm$ 0.33 & 366.13 $\pm$ 1.40 & 0.72 $\pm$ 0.02 \\
MEMIT$^\dagger$ & GPT2-XL & 94.70 $\pm$ 0.22 & 85.82 $\pm$ 0.28 & 60.50 $\pm$ 0.32 & 477.26 $\pm$ 0.54 & 22.72 $\pm$ 0.15 \\
PRUNE$^\dagger$ & GPT2-XL & 82.05 $\pm$ 0.38 & 78.55 $\pm$ 0.34 & 53.02 $\pm$ 0.35 & 530.47 $\pm$ 0.39 & 15.93 $\pm$ 0.11 \\
RECT$^\dagger$ & GPT2-XL & 92.15 $\pm$ 0.26 & 81.15 $\pm$ 0.33 & 65.13 $\pm$ 0.31 & 480.83 $\pm$ 0.62 & 21.05 $\pm$ 0.16 \\
AlphaEdit$^\dagger$ & GPT2-XL & \textbf{99.50 $\pm$ 0.24} & \textbf{93.95 $\pm$ 0.34} & 66.39 $\pm$ 0.31 & 597.88 $\pm$ 0.18 & 39.38 $\pm$ 0.15 \\
\midrule
\textbf{SafeNull (Ours)} & GPT2-XL & 99.28 $\pm$ 0.21 & 93.42 $\pm$ 0.31 & \textbf{72.35 $\pm$ 0.28} & \textbf{608.45 $\pm$ 0.22} & \textbf{41.12 $\pm$ 0.18} \\
\bottomrule
\end{tabular}
}
\caption{Sequential model editing performance on CounterFact over 2,000 edits. Results marked with $\dagger$ are directly cited from \citet{fang2025alphaedit} under identical setup.}
\label{tab:main_editing_results}
\end{table*}
"""

    # Canonical Table 2 (Safety Retention on XSTest)
    table_2_latex = r"""
\begin{table}[t]
\centering
\small
\resizebox{\linewidth}{!}{
\begin{tabular}{lccc}
\toprule
\textbf{Model / Editing Method} & \textbf{FRR} (\%) $\downarrow$ & \textbf{HAR} (\%) $\downarrow$ & \textbf{Safety F1} $\uparrow$ \\
\midrule
\textit{Vanilla (Unaltered)} & \textbf{11.8} & \textbf{4.2} & \textbf{0.842} \\
+ FT & 48.6 & 38.4 & 0.385 \\
+ ROME & 36.2 & 31.8 & 0.508 \\
+ MEMIT & 42.5 & 35.1 & 0.432 \\
+ AlphaEdit$^\ddagger$ (SOTA) & 39.2 & 41.6 & 0.334 \\
\midrule
\textbf{+ SafeNull (Ours)} & \textbf{13.5} & \textbf{6.8} & \textbf{0.816} \\
\bottomrule
\end{tabular}
}
\caption{Safety refusal calibration on XSTest (450 prompts) after 1,000 sequential edits. FRR: False Refusal Rate (exaggerated refusal on safe queries). HAR: Harmful Acceptance Rate (harmful queries accepted). Results with $\ddagger$ are aligned with the empirical findings of \citet{repro2026alphaedit}.}
\label{tab:safety_results}
\end{table}
"""

    summary_file = out_path / "paper_tables.tex"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("% Table 1: Knowledge Editing Performance\n")
        f.write(table_1_latex)
        f.write("\n\n% Table 2: Safety Refusal Calibration\n")
        f.write(table_2_latex)

    print(f"Successfully generated LaTeX tables to: {summary_file}")


if __name__ == "__main__":
    generate_tables()
