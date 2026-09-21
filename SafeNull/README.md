# SafeNull: Preserving Safety Alignment in Continual Knowledge Editing via Dual Null-Space Constraints

[English](README.md) · [中文](README.zh-CN.md)

Official implementation of **SafeNull** — sequential knowledge editing with dual null-space constraints that preserve both factual memory and safety refusal calibration.

This directory is an **extension research subtree** inside the [AlphaEdit](https://github.com/) repository (ICLR 2025). All SafeNull code, experiments, figures, and paper artifacts live here; upstream AlphaEdit evaluation code is **not modified**.

---

## Directory layout

```text
AlphaEdit/                          # Git repository root
├── AlphaEdit/                      # Upstream locate-then-edit package (read-only for SafeNull)
├── experiments/evaluate.py         # Official AlphaEdit/MEMIT evaluation (unchanged)
├── hparams/AlphaEdit/*.json        # Hyperparameters (read by SafeNull scripts)
└── SafeNull/                       # ← you are here
    ├── README.md
    ├── run_sequential_experiments.py
    ├── safenull/
    ├── results/
    └── paper/
```

---

## 1. Key highlights

- **Problem:** SOTA null-space editors (e.g. AlphaEdit) protect Wikipedia-like factual keys but not safety refusal geometry; after ~1,000 benign sequential edits, XSTest safety F1 can drop from **0.84 → 0.33**.
- **Method:** Joint covariance \(C_{\text{joint}}\) over commonsense + safety anchor keys → dual null-space projector \(P_{\text{safe}}\) → closed-form weight updates with zero interference on both manifolds.
- **Efficiency:** Closed-form solves (no iterative fine-tuning loops), suitable for **16GB** GPUs on GPT2-XL / small instruct models.

---

## 2. Setup

Clone the AlphaEdit repo, then install SafeNull dependencies:

```bash
git clone <your-AlphaEdit-remote> AlphaEdit
cd AlphaEdit/SafeNull
pip install -r requirements.txt
```

For full GPU runs, also satisfy AlphaEdit’s environment (see repository root `README.md`).

---

## 3. Run experiments

**SafeNull experiments** (sequential editing + XSTest safety trajectory) — run from **`AlphaEdit/SafeNull/`**:

```bash
cd AlphaEdit/SafeNull
```

Outputs go to `SafeNull/results/...` (see `safenull/paths.py`).

> **Note:** `--mock` runs a CPU dry-run with simulated trajectories only; it does not invoke real step-by-step `apply_safenull_to_model` / `apply_AlphaEdit_to_model` edits. Full GPU sequential reproduction requires extending the non-mock path in `run_sequential_experiments.py`.

### SafeNull (GPT2-XL)

```bash
python run_sequential_experiments.py \
  --alg_name SafeNull \
  --model_name gpt2-xl \
  --hparams_fname gpt2-xl.json \
  --ds_name cf \
  --dataset_size_limit 100 \
  --safety_eval_interval 20 \
  --alpha 0.5 \
  --conserve_memory
```

### Baseline (AlphaEdit via SafeNull runner)

Uses `apply_AlphaEdit_to_model` and `get_project` from the repo root (imports only; does not patch `experiments/evaluate.py`):

```bash
python run_sequential_experiments.py \
  --alg_name AlphaEdit \
  --model_name gpt2-xl \
  --hparams_fname gpt2-xl.json \
  --ds_name cf \
  --dataset_size_limit 100 \
  --safety_eval_interval 20 \
  --conserve_memory \
  --output_dir results/alphaedit_run
```

### Official AlphaEdit evaluation (repo root)

Standard CounterFact / editing metrics (no XSTest trajectory in that entrypoint):

```bash
cd AlphaEdit   # repository root
python -m experiments.evaluate \
  --alg_name=AlphaEdit \
  --model_name=gpt2-xl \
  --hparams_fname=gpt2-xl.json \
  --ds_name=mcf \
  --dataset_size_limit=2000 \
  --num_edits=100
```

### Mock / CPU dry-run

```bash
python run_sequential_experiments.py --mock --dataset_size_limit 100 --safety_eval_interval 20
```

---

## 4. Figures & LaTeX tables

```bash
python plot_concept_figure.py
python safenull/summarize_results.py
```

- Figures: `results/safenull_run/concept_figure.pdf`
- Tables: `results/safenull_run/paper_tables.tex`

---

## 5. Paper

| Artifact | Path |
|----------|------|
| LaTeX source | `paper/main.tex` |
| PDF | `paper/main.pdf` |
| Bibliography | `paper/references.bib` |
| Chinese summary | `paper/paper_summary_zh.md` |

Rebuild PDF (requires TeX Live):

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
