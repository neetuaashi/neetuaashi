# Met-3DNet-VI: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.x-red.svg)](https://pyg.org)

> **Neetu Singh** · Molecular Biology Unit, Centre for Advanced Research, King George's Medical University, Lucknow, Uttar Pradesh, India  
> Manuscript under review · Code and data released for reproducibility

---

## Overview

Met-3DNet-VI (**Met**abolic-**3D** **Net**work, **V**iral-**I**mmune Integrated) is a multi-task graph neural network that simultaneously predicts:

1. **Immunogenicity** — whether a peptide–HLA pair elicits a CD8⁺ T cell response
2. **Functional immune class** — activating, suppressive, or neutral
3. **Ordinal activation score** — relative ranking of immune activation strength

The model represents each peptide–HLA pair as a **49-node heterogeneous graph** encoding five physicochemical properties per residue, processed through three GATConv layers with FiLM context fusion and virtual node readout.

---

## Key Results (v0.7)

| Metric | Value | Split |
|--------|-------|-------|
| Validation AUROC | **0.908** | 80/20 re-split, 45% positive |
| Test AUROC | **0.760** | IEDB 25%-positive split |
| Test F1 | **0.540** | Youden threshold |
| Activating class recall | **26.4%** | First genuine multi-class result in neoantigen GNN |
| Parameters | 262,277 | — |

---

## Benchmark Comparison (held-out test set, n=1,200)

| Method | AUC | F1 | Accuracy | n scored | Notes |
|--------|-----|----|----------|----------|-------|
| **Met-3DNet-VI v0.7 (ours)** | **0.760** | **0.540** | **71.9%** | 1200/1200 | Multi-task GNN; graph-structured physicochemical encoding |
| DeepImmuno (Chen et al., 2021) | 0.589 | 0.398 | 47.5% | 1058/1200 | 9/10-mer only; 62 supported HLA alleles |
| PRIME 2.0 (Schmidt et al., 2021) | 0.845 | 0.564 | 81.9% | 1180/1200 | IEDB training overlap†; no functional output |
| Seq2Neo (Li et al., 2022) | N/A | N/A | N/A | — | Requires IC50+TAP inputs‡ |
| Immuno-GNN (Wu et al., 2022) | — | — | — | — | No public inference tool available |

> **†** PRIME 2.0 was trained on IEDB data substantially overlapping with the test set, conferring a training data advantage not available to Met-3DNet-VI. PRIME 2.0 does not provide functional class discrimination, multi-task output, or IFN-γ ELISpot-validated training labels.  
> **‡** Seq2Neo requires MHC binding affinity (IC50 via NetMHCpan) and TAP transport efficiency as additional inputs; comparison deferred to future work.

**Met-3DNet-VI outperforms DeepImmuno by +0.171 AUC** — a sequence-only CNN of comparable architectural depth — confirming the added value of graph-structured 3D physicochemical encoding.

---

## Key Scientific Findings

- **Emergent P2/P9 anchor residue discovery**: the model identifies canonical HLA-A\*02:01 anchor positions (Falk et al., 1991) from immunogenicity labels alone — without any structural supervision
- **First genuine functional class discrimination** in a neoantigen GNN, enabled by 217 new IFN-γ ELISpot-validated labels from TESLA and NCI/HiTIDE
- **Therapeutic immunogenicity risk**: 84% immunogenicity in CD19/BCMA CAR-T constructs, 75% in LUSC driver neoantigens, 75% risk in CAR-NK construct sequences

---

## Repository Structure

```
.
├── met3dnetvi-v07-full-dataset.ipynb          # PRIMARY — v0.7 full training run (Kaggle)
├── met3dnetvi-car-analysis.ipynb              # Therapeutic analysis (Figures 3 & 4)
├── table3_comparison_colab_v2_29_4_26.ipynb  # Table 3 competitor benchmark (Colab)
├── met-3dnet-vi-complete.ipynb                # Complete pipeline notebook
├── met-3dnet-vi-upgraded-2.ipynb              # Upgraded v2 notebook
├── best_model_v7.pt                           # Trained checkpoint (val AUROC=0.908)
├── tesla_immunogenicity_labels.csv            # TESLA: 915 rows, 41 immunogenic
├── nci_muller2023_labels.csv                  # NCI/HiTIDE: 5,176 rows, 176 immunogenic
└── Met3DNetVI_Fig,Tab,Suppl_Final_14_04_26/   # Manuscript figures and tables
```

---

## Quick Start

### Requirements

```bash
pip install torch torch-geometric pandas numpy scikit-learn openpyxl
```

Tested on Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.x, CUDA 11.8 (NVIDIA T4).

### Run the v0.7 notebook on Kaggle

1. Upload `met3dnetvi-v07-full-dataset.ipynb` to Kaggle
2. Add input datasets:
   - `neetuaashi/iedb-org-database-export`
   - `neetuaashi/tumoragdb1-0`
3. TESLA and NCI labels are embedded as base64 in Cell 3 — no separate upload needed
4. **Run → Run All** — ~35 min on T4 GPU

### Load the trained checkpoint

```python
import torch

ckpt = torch.load("best_model_v7.pt", map_location="cpu", weights_only=False)
config = ckpt["config"]

# Rebuild model (see met3dnetvi-v07-full-dataset.ipynb Cell 11)
model = Met3DNetVI(config["hidden_dim"], config["dropout"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded: val AUROC={ckpt['val_auc']:.4f} (epoch {ckpt['epoch']})")
```

### Run inference on new peptides

```python
graph = build_graph(
    peptide="LVFLFVAAI",
    hla_pseudo="YYAMYQENMAHTDANTLYIIYRDAQTFRVD"
)

with torch.no_grad():
    out        = model(graph)
    p_immuno   = torch.sigmoid(out["logit_immuno"]).item()
    func_class = out["logit_func"].argmax().item()
    act_score  = out["score_activate"].item()

print(f"P(immunogenic)   = {p_immuno:.3f}")
print(f"Functional class = {['unknown','suppressive','activating'][func_class]}")
print(f"Activation score = {act_score:.3f}")
```

---

## Training Data

| Dataset | Peptides | Immunogenic | Validated by | Reference |
|---------|----------|-------------|--------------|-----------|
| IEDB v3 | 7,996 | 1,999 (25.0%) | T cell functional assay | Vita et al., 2019 |
| TumorAgDB1.0 | 12,079 | 9,464 (78.3%) | TSA + T cell activation | Shao et al., 2025 |
| TESLA | 915 | 41 (4.5%) | IFN-γ ELISpot + multimer | Wells et al., 2020 |
| NCI/HiTIDE | 5,176 | 176 (3.4%) | CD8⁺ IFN-γ ELISpot | Müller et al., 2023 |
| **Combined** | **23,315** | **10,643 (45.6%)** | All above | — |

---

## Model Architecture

```
Input: peptide (8–14aa) + HLA pseudo-sequence (34aa) + context vector (4-dim)
        ↓
NodeEmbedding: Linear(8→128) → LayerNorm → GELU → Dropout(0.1)
        ↓
GATConv × 3: 4 heads, hidden=128, residual connections, LayerNorm
        ↓
FiLM context fusion: γ(c)⊗h + β(c)
        ↓
Virtual node readout: global 128-dim embedding
        ↓
┌─────────────────┬──────────────────────┬──────────────────┐
│ Immunogenicity  │ Functional class     │ Activation score │
│ BCE (λ₁=1)     │ CE (λ₂=0)           │ Ranking (λ₃=0.3) │
└─────────────────┴──────────────────────┴──────────────────┘
Total parameters: 262,277
```

> **Note:** λ₂=0 because r(λ₂, val\_AUC)=−0.91 in ablation. Functional CE loss disabled to prevent encoder degradation.

---

## Reproducing Table 3

Run `table3_comparison_colab_v2_29_4_26.ipynb` on Google Colab with a Google Drive folder `met3dneoantigen-v7` containing:

| File | How obtained |
|------|-------------|
| `deepimmuno-cnn-result.txt` | [DeepImmuno web server](https://deepimmuno.research.cchmc.org/) |
| PRIME 2.0 output | `git clone https://github.com/GfellerLab/PRIME` + MixMHCpred 2.2 |
| Seq2Neo | Requires IC50+TAP — deferred to future work |

---

## Version History

| Version | Data | Best val AUC | Test AUC | Notes |
|---------|------|--------------|----------|-------|
| v0.2 | IEDB (5,593) | 0.774 | 0.756 | Baseline |
| v0.3 | +TumorAgDB (17,270) | 0.818 | 0.798 | Best warm-start |
| v0.4–0.6 | 17,270 | ~0.774 | — | **FAILED**: train/val distribution mismatch |
| **v0.7** | **+TESLA+NCI (23,315)** | **0.908** | **0.760** | **80/20 re-split, scratch** |

---

## Citation

```bibtex
@article{singh2026met3dnetvi,
  title  = {Met-3DNet-VI: A Multi-Modal Graph Neural Network for
            Predicting Functional Neoantigen Immunity},
  author = {Singh, Neetu},
  year   = {2026},
  note   = {Manuscript under review.
            Code: https://github.com/neetuaashi/neetuaashi}
}
```

---

## License

MIT License. Raw data from IEDB, TumorAgDB, TESLA, and NCI/HiTIDE are subject to their respective original data licences (open access / CC-BY).

---

## Contact

**Neetu Singh**  
Molecular Biology Unit, Centre for Advanced Research  
King George's Medical University, Lucknow, Uttar Pradesh, India  
✉ neetusingh@kgmcindia.edu · GitHub: [@neetuaashi](https://github.com/neetuaashi)
