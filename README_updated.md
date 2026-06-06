# Met-3DNet-VI/VII: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.x-orange.svg)](https://pyg.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19886437.svg)](https://doi.org/10.5281/zenodo.19886437)

**Neetu Singh** · Molecular Biology Unit, Centre for Advanced Research, King George's Medical University, Lucknow, Uttar Pradesh, India
*Manuscript under review · Code and data released for reproducibility*

---

## Overview

**Met-3DNet-VI** (GNN-only, v0.7) and **Met-3DNet-VII** (GNN+CNN hybrid, v0.8) are multi-task neural networks that simultaneously predict:

- **Immunogenicity** — whether a peptide–HLA pair elicits a CD8⁺ T cell response
- **Functional immune class** — activating, suppressive, or neutral *(exploratory)*
- **Ordinal activation score** — relative ranking of immune activation strength

Met-3DNet-VI represents each peptide–HLA pair as a **49-node heterogeneous graph** encoding five physicochemical properties per residue, processed through three GATConv layers with 10-dimensional FiLM context fusion and virtual node readout.

Met-3DNet-VII augments this with a **parallel 1D-CNN branch** for sequence motif extraction, trained jointly with a dual-checkpoint strategy that independently optimises AUROC and activating-class recall.

---

## Key Results

### Met-3DNet-VI v0.7 (GNN-only)

| Metric | Value | Split |
|---|---|---|
| Validation AUROC | 0.908 | 80/20 re-split, 45% positive |
| Test AUROC | 0.760 | IEDB 25%-positive split, n=1,200 |
| Test F1 | 0.540 | Youden threshold |
| Test Accuracy | 71.9% | — |
| Activating class recall | 26.4% | First genuine functional class result in neoantigen GNN |
| Parameters | 262,277 | — |

### Met-3DNet-VII v0.8 (GNN+CNN, λ₂=0.3, dual checkpoint)

| Checkpoint | Test AUROC | Test F1 | Accuracy | Activating Recall |
|---|---|---|---|---|
| AUC-optimal (epoch 16) | 0.877 | 0.660 | 77.6% | 15.5% |
| ActRec-optimal (epoch 40) | **0.878** | **0.668** | **79.2%** | **23.6%** |
| Val best (AUC) | 0.8861 | — | — | — |
| Val best (ActRec) | — | — | — | 34.7% |

### Benchmark Comparison (held-out test set, n=1,200)

| Method | AUC | F1 | Accuracy | n scored | Notes |
|---|---|---|---|---|---|
| **Met-3DNet-VII v0.8** *(ours)* | **0.878** | **0.668** | **79.2%** | 1200/1200 | GNN+CNN hybrid; dual checkpoint; ActRec=23.6% |
| **Met-3DNet-VI v0.7** *(ours)* | 0.760 | 0.540 | 71.9% | 1200/1200 | Multi-task GNN; graph physicochemical encoding |
| PRIME 2.0 (Schmidt et al., 2021) | 0.845 | 0.564 | 81.9% | 1180/1200 | IEDB training overlap†; no functional output |
| DeepImmuno (**Li** et al., 2021) | 0.589 | 0.398 | 47.5% | 1058/1200 | 9/10-mer only; 62 supported HLA alleles |
| Seq2Neo (**Diao** et al., 2022) | N/A | N/A | N/A | — | Requires IC50+TAP inputs‡ |
| Immuno-GNN (**Wu** et al., **2023**) | — | — | — | — | No public inference tool available |

> † PRIME 2.0 was trained on IEDB data substantially overlapping with the test set. It does not provide functional class discrimination, multi-task output, or IFN-γ ELISpot-validated training labels.
> ‡ Seq2Neo requires MHC binding affinity (IC50) and TAP transport efficiency as additional inputs; comparison deferred to future work.

---

## Key Scientific Findings

- **Emergent P2/P9 anchor residue discovery**: the model identifies canonical HLA-A*02:01 anchor positions (Falk et al., 1991) from immunogenicity labels alone — without any structural supervision
- **First genuine functional class discrimination in a neoantigen GNN**, enabled by 9,704 activating labels from TumorAgDB1.0 and 217 new IFN-γ ELISpot-validated labels from TESLA and NCI/HiTIDE
- **TumorAgDB2.0 hard-negative augmentation**: 7,024 mechanistically informative hard negatives (NetMHCpan rank < 2.0); physics baseline RF-11dim AUROC = 0.834 ± 0.062
- **External validation** on 971 experimentally confirmed neoantigens (4 independent sets); sequence-derived FiLM features discriminate without binding scores
- **CIImm as 11th FiLM dimension** (Miyazawa–Jernigan contact-pair energy): outperforms Boman index on 3/4 external sets; Pearson r(Boman, CIImm) = −0.251 (near-orthogonal)
- **Therapeutic immunogenicity risk**: 84% in CD19/BCMA CAR-T constructs, 75% in LUSC driver neoantigens, 75% in CAR-NK construct sequences

---

## Repository Structure

```
neetuaashi/
├── README.md
├── notebooks/
│   ├── met3dnetvi-v07-full-dataset.ipynb          # Met-3DNet-VI v0.7 full training (Kaggle T4)
│   ├── met3dnetvi-car-analysis.ipynb              # Therapeutic analysis (Figures 3 & 4)
│   ├── table3_comparison_colab_v2_29_4_26.ipynb  # Table 3 benchmark comparison (Colab)
│   ├── met-3dnet-vi-complete.ipynb                # Complete pipeline notebook
│   ├── met3dnet_features_validation_v3.ipynb      # TumorAgDB2.0 feature validation (20 checks)
│   └── met3dnetvi_extval_v3.ipynb                 # External validation (ITSNdb/NECID/VDJdb)
├── src/
│   ├── features_updated.py                        # FiLM feature engineering (10-dim vector)
│   ├── features_updated_patched.py                # Patched: fixes pd.NA, float-nan, double log1p
│   └── resave_model_full.py                       # Converts state_dict checkpoint → full model
├── data/
│   ├── TumorAgDB2.0/
│   │   ├── tumoragdb2_immunogenic.csv             # 101 immunogenic peptides, full annotation
│   │   ├── tumoragdb2_hard_negatives.csv          # 7,024 hard negatives (rank < 2.0)
│   │   ├── training_augmentation.csv              # 7,125 combined (101 + 7,024)
│   │   ├── pubdata_2024_2025.csv                  # 543 PubData 2024-25, 78 features
│   │   └── external_validation_set.csv            # 1,086 IEDB-independent extval set
│   └── DifferentNeoAntigen/
│       ├── extval_ITSNdb.csv                      # 199 (129 pos, 70 neg) — ELISpot confirmed
│       ├── extval_Val_dataset.csv                 # 120 (7 pos, 113 neg) — Robbins et al.
│       ├── extval_NECID_mhci.csv                  # 652 (283 pos, 369 neg) — IFN-γ ELISpot
│       ├── extval_VDJdb_TAA.csv                   # 154 all pos — VDJdb TAA/CTA score≥2
│       └── extval_DifferentNeoAntigen_combined.csv # 971 (419 pos, 552 neg)
├── tesla_immunogenicity_labels.csv
├── nci_muller2023_labels.csv
└── best_model_v7.pt                               # Training dict: state_dict + val_auc + config
```

---

## Kaggle Datasets

| Dataset | URL | Size | Contents |
|---|---|---|---|
| **TumorAgDB2.0** | [kaggle.com/datasets/neetuaashi/tumoragdb2-0](https://www.kaggle.com/datasets/neetuaashi/tumoragdb2-0) | 31 MB | Hard-negative augmentation, features_updated.py, best_model_v7.pt |
| **Different_NeoAntigen_Dataset** | [kaggle.com/datasets/neetuaashi/different-neoantigen-dataset](https://www.kaggle.com/datasets/neetuaashi/different-neoantigen-dataset) | 378 MB | ITSNdb, NECID, Val_dataset, VDJdb 2026-05-16, CIImm index, extval CSVs |

---

## Quick Start

### Requirements

```bash
pip install torch torch-geometric pandas numpy scikit-learn openpyxl matplotlib seaborn scipy
```

### Load the checkpoint

`best_model_v7.pt` is saved as a **training dictionary** (not a full model object). Two options:

```python
import torch

# Option A: Load state dict and rebuild model (requires model class)
ckpt = torch.load("best_model_v7.pt", map_location="cpu", weights_only=False)
# ckpt keys: ['state_dict', 'val_auc', 'epoch', 'config']
config = ckpt["config"]
model = Met3DNetVI(**config)          # import from your training notebook
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"val_auc={ckpt['val_auc']:.4f}  epoch={ckpt['epoch']}")

# Option B: Convert to full model first (recommended — run once on training machine)
# python src/resave_model_full.py
model = torch.load("best_model_v7_full.pt", map_location="cpu", weights_only=False)
model.eval()
```

### Run inference

```python
graph = build_graph(peptide="LVFLFVAAI", hla_pseudo="YYAMYQENMAHTDANTLYIIYRDAQTFRVD")

with torch.no_grad():
    out       = model(graph)
    p_immuno  = torch.sigmoid(out["logit_immuno"]).item()
    func_cls  = out["logit_func"].argmax().item()
    act_score = out["score_activate"].item()

print(f"P(immunogenic)   = {p_immuno:.3f}")
print(f"Functional class = {['unknown','suppressive','activating'][func_cls]}")
print(f"Activation score = {act_score:.3f}")
```

---

## Training Data

| Dataset | Peptides | Immunogenic | Validated by | Reference |
|---|---|---|---|---|
| IEDB v3 | 7,996 | 1,999 (25.0%) | T cell functional assay | Vita et al., 2019 |
| TumorAgDB1.0 | 12,079 | 9,464 (78.3%) | TSA + T cell activation | Shao et al., 2025 |
| TESLA | 915 | 41 (4.5%) | IFN-γ ELISpot + multimer | Wells et al., 2020 |
| NCI/HiTIDE | 5,176 | 176 (3.4%) | CD8⁺ IFN-γ ELISpot | Müller et al., 2023 |
| **TumorAgDB2.0 (augmentation)** | **7,125** | **101** | **ELISpot; NetMHCpan<2** | **Shao et al., 2025** |
| **Combined (Met-3DNet-VI)** | **23,315** | **10,643 (45.6%)** | All above | — |

---

## Model Architecture

### Met-3DNet-VI (v0.7 — GNN-only)

```
Input: peptide (8–14 aa) + HLA pseudo-sequence (34 aa) + 10-dim FiLM context vector
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
│ BCE (λ₁=1.0)   │ CE (λ₂=0.0)         │ Ranking (λ₃=0.3) │
└─────────────────┴──────────────────────┴──────────────────┘
Total parameters: 262,277
Note: λ₂=0 because r(λ₂, val_AUC)=−0.91 in ablation (Table 4).
```

### Met-3DNet-VII (v0.8 — GNN+CNN, λ₂=0.3)

```
Input: same as above
        ↓
┌──────────────────────────────┬──────────────────────────────┐
│  GNN Branch (GATConv × 3     │  CNN Branch (1D-Conv × 3)    │
│  + FiLM + virtual readout)   │  → max-pool → 64-dim         │
└──────────────────────────────┴──────────────────────────────┘
        ↓ concat → joint embedding
BCE(λ₁) + CE(λ₂=0.3) + Ranking(λ₃=0.3)
Dual checkpoint: AUC-optimal (epoch 16) + ActRec-optimal (epoch 40)
```

### FiLM Conditioning Vector (11 dimensions)

| Dim | Feature | Source | AUROC (TumorAgDB2.0) |
|---|---|---|---|
| 0 | NetMHCpan rank | NetMHCpan 4.1 | 0.790 |
| 1 | TAP score | Sette TAP | 0.526 |
| 2 | NetChop score | NetChop 3.1 | 0.500 |
| 3 | log(1+TPM) | RNA-seq | 0.678 |
| 4 | DAI | NetMHCpan | 0.626 |
| 5 | Boman index | Boman 2003 | 0.509 |
| 6 | Aliphatic idx/100 | Ikai 1980 | 0.534 |
| 7 | GRAVY | Kyte-Doolittle | 0.518 |
| 8 | Net charge/5 | pH 7.0 | 0.534 |
| 9 | Length norm | Sequence | 0.644 |
| 10 | **Mean CIImm** *(exploratory)* | **Miyazawa-Jernigan** | **0.574** |

> Dims 0–4 require per-peptide binding scores; prior values used when absent (AUROC=0.500 without scores).

---

## External Validation

| Dataset | n | RF-11dim AUROC | Best feature | Benchmark |
|---|---|---|---|---|
| ITSNdb | 199 | 0.682 ± 0.075 | net_charge/5 (0.720) | **Fairest** (homogeneous 9–10-mer, 31 HLA alleles) |
| Val_dataset | 120 | 0.709 ± 0.144 ⚠ | net_charge/5 (0.590) | Unreliable (n_pos=7 only) |
| NECID_mhci | 652 | 0.874 ± 0.009 | length_norm (0.669) | Inflated by 8–14 aa length range |
| Combined | 971 | 0.927 ± 0.006 | length_norm (0.703) | NECID-dominated |

GNN target on ITSNdb: **> 0.702** to justify graph architecture over RF physics baseline.

---

## Version History

| Version | Model | n | Val AUROC | Test AUROC | ActRec | Notes |
|---|---|---|---|---|---|---|
| v0.2 | GNN-only | 5,593 | 0.774 | 0.756 | 0% | Baseline |
| v0.3 | GNN-only | 17,270 | 0.818 | 0.798 | 0% | Best warm-start |
| v0.4–0.6 | GNN-only | 17,270 | ~0.774 | — | 0% | FAILED: train/val mismatch |
| **v0.7** | **GNN-only** | **23,315** | **0.908** | **0.760** | **26.4%** | **80/20 re-split** |
| **v0.8** | **GNN+CNN** | **26,405** | **0.8861** | **0.878** | **23.6%** | **λ₂=0.3, dual checkpoint** |

---

## Citation

```bibtex
@article{singh2026met3dnetvi,
  title  = {Met-3DNet-{VI}/{VII}: A Multi-Modal Graph Neural Network for
            Predicting Functional Neoantigen Immunity},
  author = {Singh, Neetu},
  year   = {2026},
  note   = {Manuscript under review.
            Code: https://github.com/neetuaashi/neetuaashi
            Data: https://doi.org/10.5281/zenodo.19886437}
}
```

---

## Data Availability

| Resource | URL |
|---|---|
| Code + notebooks | [github.com/neetuaashi/neetuaashi](https://github.com/neetuaashi/neetuaashi) |
| Model + large datasets | [doi.org/10.5281/zenodo.19886437](https://doi.org/10.5281/zenodo.19886437) |
| TumorAgDB2.0 (Kaggle) | [kaggle.com/datasets/neetuaashi/tumoragdb2-0](https://www.kaggle.com/datasets/neetuaashi/tumoragdb2-0) |
| Different_NeoAntigen_Dataset (Kaggle) | [kaggle.com/datasets/neetuaashi/different-neoantigen-dataset](https://www.kaggle.com/datasets/neetuaashi/different-neoantigen-dataset) |
| IEDB v3 | [iedb.org/database_export_v3.php](https://www.iedb.org/database_export_v3.php) |
| TumorAgDB1.0 | [tumoragdb.com.cn](http://tumoragdb.com.cn) |
| TESLA | [Synapse syn21048999](https://www.synapse.org/#!Synapse:syn21048999) |
| NCI/HiTIDE | [figshare.com/s/147e67dde683fb769908](https://figshare.com/s/147e67dde683fb769908) |
| VDJdb 2026-05-16 | [github.com/antigenomics/vdjdb-db/releases/tag/2026-05-16](https://github.com/antigenomics/vdjdb-db/releases/tag/2026-05-16) |

---

## License

MIT License. Raw data from IEDB, TumorAgDB, TESLA, NCI/HiTIDE, ITSNdb, NECID, and VDJdb are subject to their respective original data licences (open access / CC-BY). VDJdb: Bagaev et al., Nat Methods 2020.

---

## Contact

**Neetu Singh**
Molecular Biology Unit, Centre for Advanced Research
King George's Medical University, Lucknow, Uttar Pradesh, India
✉ neetusingh@kgmcindia.edu · GitHub: [@neetuaashi](https://github.com/neetuaashi)
