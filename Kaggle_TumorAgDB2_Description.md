## TumorAgDB2.0 — Met-3DNet-VI/VII Training Augmentation Dataset

Curated from **TumorAgDB2.0** (Shao et al., 2025; tumoragdb.com.cn; 187,223 entries) for training augmentation and external validation of the Met-3DNet-VI/VII neoantigen immunogenicity prediction framework. Published as companion data to: *Singh N. Met-3DNet-VI/VII: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity. Manuscript under review, 2026.*

### Files

| File | Rows | Description |
|---|---|---|
| `tumoragdb2_immunogenic.csv` | 101 | Immunogenic peptides (NCI:75, TESLA:15, HiTIDE:11), 8–14 aa, full 10-dim feature annotation |
| `tumoragdb2_hard_negatives.csv` | 7,024 | Hard negatives: NetMHCpan 4.1 rank < 2.0 — high-affinity binders that fail T cell recognition |
| `training_augmentation.csv` | 7,125 | Combined set: 101 immunogenic + 7,024 hard negatives |
| `pubdata_2024_2025.csv` | 543 | PubData 2024–2025 immunogenic peptides with 78 pre-computed physicochemical features |
| `external_validation_set.csv` | 1,086 | 543 PubData + 543 matched hard negatives — fully independent of IEDB training data |
| `features_updated.py` | — | FiLM feature engineering module: 10-dimensional context vector (NetMHCpan rank, TAP, NetChop, log-TPM, DAI, Boman, Aliphatic, GRAVY, Net charge, Length) |
| `features_updated_patched.py` | — | Production version — fixes 3 bugs: pd.NA TypeError, float-nan bypass, double log1p |
| `best_model_v7.pt` | — | Met-3DNet-VI v0.7 checkpoint (training dict: state_dict + val_auc=0.908 + epoch + config). See resave_model_full.py to convert to full model for inference. |
| `NeoAntigen-PubData 2024-2025.xlsx` | 1,086 rows | Raw PubData source with 78 pre-computed physicochemical features (boman_index, hmoment, aindex, VHSE1-8, protFP1-8, Kidera factors 1-10) |
| `Non-immunogenic Neo-peptide Dataset_less_than_2.xlsx` | 36,589 | Full source of hard negatives; filter `mutant_rank_netMHCpan < 2.0` to obtain the 7,024-row subset |
| `immunogenic Neo-peptide Dataset.xlsx` | 101 | Raw immunogenic source file |

### Key Statistics

- **Hard negative definition:** NetMHCpan 4.1 rank < 2.0 (confirmed high-affinity binders with no T cell response — the mechanistically hardest negatives)
- **Physics baseline (Random Forest, 5-fold CV, 11-dim FiLM):** AUROC = 0.834 ± 0.062
- **GNN target:** Must exceed **0.854** (RF + 0.02 margin) to justify graph architecture
- **Boman index validated:** Pearson r = 0.9461 vs PubData pre-computed values (p = 3.82×10⁻²⁶⁷)

### Feature AUROC on Hard-Negative Benchmark (n=7,125)

| Feature (dim) | AUROC | Type |
|---|---|---|
| NetMHCpan rank (0) | 0.790 | Binding-derived |
| log(1+TPM) (3) | 0.678 | Binding-derived |
| DAI (4) | 0.626 | Binding-derived |
| Length norm (9) | 0.644 | Sequence-derived |
| Boman index (5) | 0.509 | Sequence-derived |
| Mean CIImm (10) | 0.574 | Sequence-derived |

### How to Use

```python
# On Kaggle, add this dataset and run:
import pandas as pd, importlib.util

train = pd.read_csv("/kaggle/input/tumoragdb2-0/training_augmentation.csv")
print(f"Training augmentation: {len(train)} rows, {train['label'].value_counts().to_dict()}")

# Load feature engineering module
spec = importlib.util.spec_from_file_location(
    "fu", "/kaggle/input/tumoragdb2-0/features_updated_patched.py")
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)
```

### Citation

```bibtex
@article{singh2026met3dnetvi,
  title  = {Met-3DNet-{VI}/{VII}: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity},
  author = {Singh, Neetu},
  year   = {2026},
  note   = {Code: https://github.com/neetuaashi/neetuaashi  Data: https://doi.org/10.5281/zenodo.19886437}
}
```

**Original TumorAgDB2.0 source:** Shao Y, Gao Y, Wu LY, Ge SG, Wen PB. TumorAgDB1.0: tumor neoantigen database platform. Database (Oxford). 2025;2025:baaf010. doi:10.1093/database/baaf010.
