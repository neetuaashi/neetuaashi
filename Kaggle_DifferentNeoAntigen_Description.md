## Different_NeoAntigen_Dataset — External Validation for Neoantigen Immunogenicity Prediction

Five independent external validation sets and source databases for neoantigen immunogenicity prediction, used to validate Met-3DNet-VI/VII features on datasets **entirely independent of IEDB training data**. Published as companion data to: *Singh N. Met-3DNet-VI/VII: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity. Manuscript under review, 2026.*

### Files

| File | Rows | Pos | Neg | Label type |
|---|---|---|---|---|
| `extval_ITSNdb.csv` | 199 | 129 | 70 | ELISpot/multimer confirmed, 31 HLA alleles, 16 cancer types |
| `extval_Val_dataset.csv` | 120 | 7 | 113 | Robbins et al. confirmed negatives + Ehx/Yang positives |
| `extval_NECID_mhci.csv` | 652 | 283 | 369 | IFN-γ ELISpot, 8–14 aa, 23 cancer types, 69 HLA alleles |
| `extval_VDJdb_TAA.csv` | 154 | 154 | 0 | VDJdb 2026-05-16 TAA/CTA, score ≥ 2 (all confirmed T cell epitopes) |
| `extval_DifferentNeoAntigen_combined.csv` | 971 | 419 | 552 | ITSNdb + Val_dataset + NECID merged |
| `CIImm.xlsx` | 20 | — | — | Miyazawa–Jernigan contact immunogenicity index per amino acid (2 sheets: AAscores, PositionsScores) |
| `ITSNdb.csv` | 199 | — | — | Raw ITSNdb source: mutation position, WT peptide, cancer type, HLA, NeoType |
| `NECID_Query.csv` | 15,912 | — | — | Full NECID database: 44 columns, CDR3B, TCR data, assay type, clinical metadata |
| `Val_dataset.csv` | 120 | — | — | Raw Val_dataset: author, origin annotations |
| `TNB_dataset.csv` | 99,480 | — | — | ⚠ Clinical response data (DCB/NR labels) — NOT immunogenicity assay, do not use as extval |
| `vdjdb-2026-05-16/` | 146,055 | — | — | Full VDJdb 2026-05-16 release: vdjdb.txt, vdjdb.slim.txt, vdjdb_full.txt, cluster_members.txt, motif_pwms.txt |

### Key External Validation Results (11-dim FiLM, RF 5-fold CV)

| Dataset | n | RF-11dim AUROC | σ | Best sequence feature | Note |
|---|---|---|---|---|---|
| ITSNdb | 199 | 0.682 | 0.075 | net_charge/5 (0.720) | **Fairest benchmark** — 9–10-mer, diverse HLA |
| Val_dataset | 120 | 0.709 | 0.144 ⚠ | net_charge/5 (0.590) | Unreliable — only 7 positives |
| NECID_mhci | 652 | 0.874 | 0.009 | length_norm (0.669) | Partially inflated by 8–14 aa range |
| Combined | 971 | 0.927 | 0.006 | length_norm (0.703) | NECID-dominated |

**GNN target:** Must exceed RF + 0.02 = **0.702** on ITSNdb (fairest benchmark) to justify graph architecture over plain feature vector.

### CIImm (11th FiLM Dimension)

- **Source:** Miyazawa–Jernigan pairwise residue contact energy (kcal/mol)
- **Range:** −0.700 (K) to +0.719 (W)
- **vs Boman index:** Pearson r = −0.251 (p = 3.45×10⁻⁴) — near-orthogonal, non-redundant
- **Performance:** Outperforms Boman on 3/4 external validation sets (NECID Δ=+0.081, Combined Δ=+0.079)
- **Binding dims 0–4:** Return AUROC = 0.500 on all external sets (no per-peptide binding scores in these datasets)

### Why TNB_dataset is excluded from validation

`TNB_dataset.csv` contains **clinical response labels** (DCB/NR from immunotherapy cohorts), not T cell immunogenicity assay results. Clinical response is a distal outcome influenced by tumour mutational burden, immune checkpoint expression, and treatment response — not a direct measure of per-peptide T cell immunogenicity. Using it as an immunogenicity validation set would introduce systematic confounding.

### VDJdb Attribution

VDJdb 2026-05-16 release is included for research use. Please cite:
- Bagaev et al. VDJdb in the pandemic era: new observations and branch to human repertoire response to SARS-CoV-2. Nucleic Acids Res. 2022;50(D1):D1229-D1237.
- Dmitriev et al. VDJdb 2024: additions, corrected annotations and new analysis features. Nucleic Acids Res. 2024;52(D1):D1068-D1074.

Original: https://github.com/antigenomics/vdjdb-db

### Citation

```bibtex
@article{singh2026met3dnetvi,
  title  = {Met-3DNet-{VI}/{VII}: A Multi-Modal Graph Neural Network for Predicting Functional Neoantigen Immunity},
  author = {Singh, Neetu},
  year   = {2026},
  note   = {Code: https://github.com/neetuaashi/neetuaashi  Data: https://doi.org/10.5281/zenodo.19886437}
}
```
