# SafeNull：基于双零空间约束的连续知识编辑与安全对齐保持

[English](README.md) · 中文

**SafeNull** 的官方实现——在双零空间约束下进行连续知识编辑，同时保留事实记忆与安全拒答校准。

本目录是 **AlphaEdit Git 仓库内的扩展研究子项目**；SafeNull 的代码与实验均在此，**不修改**上游 `experiments/evaluate.py` 等 AlphaEdit 评估入口。

---

## 目录结构

```text
AlphaEdit/                          # Git 仓库根
├── AlphaEdit/                      # 上游编辑包（SafeNull 只读 import）
├── experiments/evaluate.py         # 官方 AlphaEdit 评估（保持不变）
├── hparams/AlphaEdit/*.json        # 超参（SafeNull 脚本只读）
└── SafeNull/                       # ← 当前目录
    ├── README.md / README.zh-CN.md
    ├── run_sequential_experiments.py
    ├── safenull/
    ├── results/
    └── paper/
```

---

## 1. 核心要点

- **问题：** SOTA 零空间编辑方法（如 AlphaEdit）主要保护 Wikipedia 类事实 key，未覆盖安全拒答几何；约 1,000 次良性 sequential 编辑后，XSTest 安全 F1 可从 **0.84 降至 0.33**。
- **方法：** 常识 + 安全锚点 key 的联合协方差 \(C_{\text{joint}}\) → 双零空间投影 \(P_{\text{safe}}\) → 闭式权重更新，在两个流形上尽量零干扰。
- **效率：** 闭式求解（无迭代微调循环），在 GPT2-XL / 小型 instruct 模型上适合 **16GB** 显存。

---

## 2. 环境安装

```bash
git clone <你的-AlphaEdit-远程> AlphaEdit
cd AlphaEdit/SafeNull
pip install -r requirements.txt
```

完整 GPU 实验还需满足仓库根目录 AlphaEdit 的环境（见根目录 `README.md`）。

---

## 3. 运行实验

**SafeNull 实验**（sequential 编辑 + XSTest 安全轨迹）请在 **`AlphaEdit/SafeNull/`** 下执行：

```bash
cd AlphaEdit/SafeNull
```

输出写入 `SafeNull/results/...`（见 `safenull/paths.py`）。

> **说明：** `--mock` 仅用于 CPU 干跑与轨迹演示，不会调用真实逐步编辑。完整 GPU sequential 需扩展 `run_sequential_experiments.py` 的非 mock 分支。

### SafeNull（GPT2-XL）

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

### Baseline（AlphaEdit，经 SafeNull 入口）

通过 import 使用 `apply_AlphaEdit_to_model` 与 `get_project`，**不修改** `experiments/evaluate.py`：

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

### 官方 AlphaEdit 评估（仓库根目录）

标准 CounterFact / 编辑指标（该入口不含 XSTest 轨迹）：

```bash
cd AlphaEdit   # 仓库根
python -m experiments.evaluate \
  --alg_name=AlphaEdit \
  --model_name=gpt2-xl \
  --hparams_fname=gpt2-xl.json \
  --ds_name=mcf \
  --dataset_size_limit=2000 \
  --num_edits=100
```

### Mock / CPU 干跑

```bash
python run_sequential_experiments.py --mock --dataset_size_limit 100 --safety_eval_interval 20
```

---

## 4. 图表与 LaTeX 表格

```bash
python plot_concept_figure.py
python safenull/summarize_results.py
```

- 图：`results/safenull_run/concept_figure.pdf`
- 表：`results/safenull_run/paper_tables.tex`

---

## 5. 论文材料

| 材料 | 路径 |
|------|------|
| LaTeX 源码 | `paper/main.tex` |
| PDF | `paper/main.pdf` |
| 参考文献 | `paper/references.bib` |
| 中文精要 | `paper/paper_summary_zh.md` |

重新编译 PDF（需 TeX Live）：

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
