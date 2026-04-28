# Met-3DNet-VI: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.x-red.svg)](https://pyg.org)

> **Neetu Singh** · Independent Researcher · April 2026  
> Manuscript under review · Code and data released for reproducibility

---

## Overview

Met-3DNet-VI (**Met**abolic-**3D** **Net**work, **V**iral-**I**mmune Integrated) is a multi-task graph neural network that simultaneously predicts:

1. **Immunogenicity** — whether a peptide–HLA pair elicits a CD8⁺ T cell response
2. **Functional immune class** — activating, suppressive, or neutral
3. **Ordinal activation score** — relative ranking of immune activation strength

The model represents each peptide–HLA pair as a **49-node heterogeneous graph** encoding five physicochemical properties per residue, processed through three GATConv layers with FiLM context fusion and virtual node readout.

### Key Results (v0.7)

| Metric | Value | Split |
|--------|-------|-------|
| Validation AUROC | **0.908** | 80/20 re-split, 45% positive |
| Test AUROC | **0.760** | IEDB 25%-positive split |
| Test F1 | 0.540 | Youden threshold |
| Activating class recall | **26.4%** | First genuine multi-class result in neoantigen GNN |
| Parameters | 262,277 | — |

### Key Scientific Findings

- **Emergent P2/P9 anchor residue discovery**: the model identifies canonical HLA-A\*02:01 anchor positions (Falk et al., 1991) from immunogenicity labels alone — without any structural supervision
- **First genuine functional class discrimination** in a neoantigen GNN, enabled by 217 new IFN-γ ELISpot-validated labels from TESLA and NCI/HiTIDE
- **Therapeutic immunogenicity risk**: 84% immunogenicity in CD19/BCMA CAR-T constructs, 75% in LUSC neoantigens, 75% risk in CAR-NK construct sequences

---

## Repository Structure

```
.
├── notebooks/
│   ├── met3dnetvi-v07-full-dataset.ipynb   # PRIMARY — v0.7 full training run
│   ├── met3dnetvi-car-analysis.ipynb        # Therapeutic analysis (Figures 3 & 4)
│   └── archive/                             # Earlier model versions (v0.2–v0.6)
├── models/
│   ├── best_model_v7.pt                     # Trained checkpoint (val AUROC=0.908)
│   └── training_history_v7.json            # Full 100-epoch training log
├── figures/
│   ├── Figure1_Training_Curves_v07.png
│   ├── Figure2_Residue_Importance_Heatmap.png
│   ├── Figure3_Immunogenicity_Landscape.png
│   └── Figure4_Activation_Suppression_Axis.png
├── manuscript/
│   ├── Met3DNetVI_Manuscript_Submission_v2.docx
│   ├── Table1_Dataset_Composition.docx
│   ├── Table2_Test_Performance.docx
│   ├── Table3_Benchmark_Comparison.docx
│   └── Supplementary_Data.docx
├── data/
│   ├── tesla_immunogenicity_labels.csv      # TESLA: 915 rows, 41 immunogenic
│   ├── nci_muller2023_labels.csv            # NCI/HiTIDE: 5,176 rows, 176 immunogenic
│   └── README_data.md                       # Data sources and download links
└── README.md
```

---

## Quick Start

### Requirements

```bash
pip install torch torch-geometric pandas numpy scikit-learn openpyxl
```

Tested on Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.x, CUDA 11.8 (NVIDIA T4).

### Run the v0.7 notebook

The primary notebook `notebooks/met3dnetvi-v07-full-dataset.ipynb` was developed and trained on [Kaggle](https://www.kaggle.com). To reproduce:

1. Upload to Kaggle and add the following input datasets:
   - `neetuaashi/iedb-org-database-export` (IEDB v3 processed splits)
   - `neetuaashi/tumoragdb1-0` (TumorAgDB1.0)
2. The TESLA and NCI labels are embedded as base64 directly in Cell 3 — no separate upload needed
3. Run all cells in sequence (Runtime → Run All)
4. Training takes approximately 35 minutes on Kaggle T4 GPU (100 epochs × ~21s/epoch)

### Load the trained checkpoint

```python
import torch
from torch_geometric.loader import DataLoader

# Load checkpoint
ckpt = torch.load("models/best_model_v7.pt", map_location="cpu", weights_only=False)
config = ckpt["config"]
# config: hidden_dim=128, dropout=0.1, lr=1e-3, lambda2=0.0, lambda3=0.3

# Rebuild model (see notebook Cell 11 for Met3DNetVI class definition)
model = Met3DNetVI(config["hidden_dim"], config["dropout"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint: val AUROC={ckpt['val_auc']:.4f} (epoch {ckpt['epoch']})")
```

### Run inference on new peptides

```python
# Build graph for a peptide–HLA pair (see notebook for full build_graph() function)
graph = build_graph(peptide="LVFLFVAAI", hla_pseudo="YYAMYQENMAHTDANTLYIIYRDAQTFRVD")

with torch.no_grad():
    out = model(graph)
    p_immuno = torch.sigmoid(out["logit_immuno"]).item()
    func_class = out["logit_func"].argmax().item()  # 0=unknown, 1=suppressive, 2=activating
    act_score  = out["score_activate"].item()

print(f"P(immunogenic) = {p_immuno:.3f}")
print(f"Functional class = {['unknown','suppressive','activating'][func_class]}")
```

---

## Training Data

The v0.7 model was trained on 23,315 peptide–HLA pairs from four public sources:

| Dataset | Peptides | Immunogenic | Validated by | Reference |
|---------|----------|-------------|--------------|-----------|
| IEDB v3 | 7,996 | 1,999 (25.0%) | T cell assay | [Vita et al., 2019](https://doi.org/10.1093/nar/gky1006) |
| TumorAgDB1.0 | 12,079 | 9,464 (78.3%) | TSA + T cell activation | [Shao et al., 2025](https://doi.org/10.1093/database/baaf010) |
| TESLA | 915 | 41 (4.5%) | IFN-γ ELISpot + multimer | [Wells et al., 2020](https://doi.org/10.1016/j.cell.2020.09.015) |
| NCI/HiTIDE | 5,176 | 176 (3.4%) | CD8⁺ IFN-γ ELISpot | [Müller et al., 2023](https://doi.org/10.1016/j.immuni.2023.09.002) |
| **Combined** | **23,315** | **10,643 (45.6%)** | All above | — |

Raw data download links are in [`data/README_data.md`](data/README_data.md).

---

## Model Architecture

```
Input: peptide (8–14aa) + HLA pseudo-sequence (34aa) + context vector (4-dim)
        ↓
NodeEmbedding: Linear(8→128) → LayerNorm → GELU → Dropout(0.1)
        ↓
GATConv × 3: 4 heads, hidden=128, residual connections, LayerNorm
        ↓
FiLM context fusion: γ(c)⊗h + β(c)  [conditioned on peptide physicochemistry]
        ↓
Virtual node readout: global 128-dim embedding
        ↓
┌─────────────────┬──────────────────────┬──────────────────┐
│ Immunogenicity  │ Functional class     │ Activation score │
│ BCE loss (λ₁=1) │ CE loss (λ₂=0)      │ Ranking (λ₃=0.3) │
│ σ(logit) → P   │ softmax → 3 classes  │ ordinal score    │
└─────────────────┴──────────────────────┴──────────────────┘
Total parameters: 262,277
```

**Critical training note**: λ₂=0 because r(λ₂, val_AUC)=−0.91 in ablation. The functional CE loss was disabled to prevent encoder degradation.

---

## Version History

| Version | Data | Best val AUC | Test AUC | Notes |
|---------|------|--------------|----------|-------|
| v0.2 | IEDB (5,593) | 0.774 | 0.756 | Baseline |
| v0.3 | +TumorAgDB (17,270) | 0.818 | 0.798 | Best warm-start |
| v0.4–0.6 | 17,270 | ~0.774 | — | FAILED: train/val distribution mismatch |
| **v0.7** | **+TESLA+NCI (23,315)** | **0.908** | **0.760** | **80/20 re-split, scratch training** |

The v0.4–0.6 failures were caused by train–validation distribution mismatch (train set 45–60% positive; original IEDB val set 25% positive). Resolved by 80/20 peptide-level re-split of the combined dataset.

---

## Citation

If you use Met-3DNet-VI in your research, please cite:

```bibtex
@article{singh2026met3dnetvi,
  title   = {Met-3DNet-VI: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity},
  author  = {Singh, Neetu},
  year    = {2026},
  note    = {Manuscript under review. Code: https://github.com/neetuaashi/neetuaashi}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Raw data from IEDB, TumorAgDB, TESLA, and NCI/HiTIDE are subject to their respective original data licences (open access / CC-BY). See [`data/README_data.md`](data/README_data.md) for details.

---

## Contact

Neetu Singh · Independent Researcher  
GitHub: [@neetuaashi](https://github.com/neetuaashi)
