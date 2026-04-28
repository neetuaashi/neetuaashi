# Data Sources for Met-3DNet-VI

This file describes all data sources used in training Met-3DNet-VI v0.7.
No raw patient-level data is redistributed here. All sources are publicly available.

---

## Files in this directory

| File | Rows | Immunogenic | Source |
|------|------|-------------|--------|
| `tesla_immunogenicity_labels.csv` | 915 | 41 | Wells et al., 2020 |
| `nci_muller2023_labels.csv` | 5,176 | 176 | Müller et al., 2023 |

These two CSV files were extracted from the original supplementary data and
reformatted for direct use in the training pipeline. Column schema:

```
peptide          str    Amino acid sequence (8–14aa)
hla_allele       str    HLA allele in format HLA-A*02:01
immunogenicity   int    1 = immunogenic, 0 = non-immunogenic
functional_class int    0 = unknown, 1 = suppressive, 2 = activating
source           str    Dataset name (TESLA or NCI_HiTIDE)
```

---

## Raw Data Download Links

### IEDB v3
- **URL**: https://iedb.org/database_export_v3.php
- **Files used**: `tcr_full_v3.zip`, `antigen_full_v3.zip`, `mhc_3d_assays.zip`
- **Reference**: Vita R et al. Nucleic Acids Research. 2019;47(D1):D339–D343.
- **Licence**: Open access (Creative Commons Attribution)

### TumorAgDB1.0
- **URL**: http://tumoragdb.com.cn/download
- **Files used**: T-mTSA-positive, Validated Immunogenic Neoantigen Data, T cell activation experiments
- **Reference**: Shao Y et al. Database (Oxford). 2025;2025:baaf010.
- **Licence**: Open access for research use

### TESLA Dataset
- **URL**: https://www.synapse.org/#!Synapse:syn21048999
- **Files used**: Supplementary Table S4 (mmc4.xlsx), Supplementary Table S7 (mmc7.xlsx)
- **Reference**: Wells DK et al. Cell. 2020;183(3):818–834.
- **Licence**: Open data use agreement (Synapse)

### NCI/HiTIDE Harmonised Dataset
- **URL**: https://figshare.com/s/147e67dde683fb769908
- **Files used**: `Neopep_data_org_txt.zip`, `HLA_allotypes.txt`
- **Reference**: Müller M et al. Immunity. 2023;56(11):2650–2663.
- **Licence**: Creative Commons Attribution (CC-BY)

---

## Processed Kaggle Datasets

The training pipeline reads from two Kaggle datasets:

- `neetuaashi/iedb-org-database-export` — processed IEDB parquet splits
- `neetuaashi/tumoragdb1-0` — TumorAgDB processed files

TESLA and NCI/HiTIDE labels are embedded as base64 in the notebook itself
(Cell 3) and written to `/kaggle/working/` at runtime — no separate dataset upload needed.

---

## Ethics and Data Redistribution

All data were sourced from publicly available databases under open licences.
No new patient data were collected. No patient-identifiable information is
redistributed in this repository. For TESLA and NCI/HiTIDE data, the
original Synapse and figshare licences apply.
