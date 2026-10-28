# Stage-4 technical report: Zenodo deposit

**Status: PUBLISHED 2026-10-28.**

| Field | Value |
| :--- | :--- |
| Version DOI | [10.5281/zenodo.23696829](https://doi.org/10.5281/zenodo.23696829) |
| Concept DOI | [10.5281/zenodo.23696828](https://doi.org/10.5281/zenodo.23696828) |
| Record | <https://zenodo.org/record/23696829> |
| Type | publication / report |
| Licence | CC-BY-4.0 |
| Version | v1.0 |
| Files | `BioSNN-Plug_Stage4_Technical_Report_v1.0.pdf` (199,443 B, md5 `28176c08...`) and `stage4_technical_report.md` (16,063 B, md5 `17c4023d...`) |

The attached Markdown is the pre-deposit snapshot of [`README.md`](README.md); the repository copy
is the source of truth for later edits. **Post-publication (2026-10-28):** e2e v10 adds a
shrunk-covariance gate that lifts the s=2 abstention from 0.720 to 0.828 (coverage 0.790,
AUROC 0.894); v11 then shows K=8 WM can be fixed (snapshot85 R² 0.9936, longtau 0.90145) but shrunk-gate
coverage remains below 0.78; v12 switches K=8 to a diag gate and both arms pass (coverage
0.782/0.805). This v1.0 record is the pre-v10/v11/v12 snapshot. Later work (all dated 2026-10-28) adds:
e2e v14 (18-cell matrix, 14/18 state-level), v15 (s=2 arms still 0/4), v16 (input-scale gate,
18/18, input-level only); spiking WM v6–v9 (v9 spike trace 0.9269/0.9276, non-trivial arm diff
0.130; v8 was a trivial solution and is not evidence); episodic v5 (local delta 0.805 at K=8,
0.922 at K=4); metacognitive v8 (reconstruction-consistency gate: quantized abstain 0.898, AUROC
0.925, coverage 0.810); and P3 v3–v5 (open-loop calibrated termination 0.999996 AUROC, FP
0.000504, hit 0.998; rolling-horizon closed-loop v6 AUROC 1.0, FP 0.000912, hit 1.0; the 500-step
pure-imagination loop remains 0.63–0.66 and open; state-level E2E v18 is 18/18). A v1.1 deposit, if
made, should carry those updates; the DOI above stays the v1.0 snapshot. The PDF was rendered from that Markdown outside the
repository (PDFs are not committed), with an ASCII-normalized text layer so extraction is clean.

The record was verified from Zenodo's **public** API after publishing: DOI, concept DOI, title,
resource type Report, licence CC-BY-4.0, version v1.0, and both attachments md5-match the local
files byte-for-byte. Publishing was done through the Zenodo API after the user explicitly requested
the deposit.
