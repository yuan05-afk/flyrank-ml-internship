# ML-12 — Tell the Story (cuts)

Companion to the deployed paper and `work/notebooks/capstone.ipynb` closing cells.

## 5-minute demo outline

1. **0:00–0:30** Decision: which page to refresh first; cost of wrong call.
2. **0:30–1:30** Data contract + leakage rule (no `trend_*` features).
3. **1:30–3:00** Baseline vs RF on client-holdout Precision@50 with base rate (~0.54).
4. **3:00–4:30** Top-3 queue rows: action + reason codes; what would make them wrong.
5. **4:30–5:00** Limits + paper URL.

## Social-post cut

Measured on 30k anonymized pages: a transparent refresh rule hit Precision@50 ≈ 0.24; a Random Forest on the same client-holdout hit ≈ 0.74 (base rate ≈ 0.54). Decision-support queue for editors — not a Google predictor. Paper + repo linked.

## Employer-facing summary

Built a content refresh ranking system on FlyRank's anonymized search starter dataset (30k pages, client-holdout validation). Compared a transparent baseline to RF/LR/DT; selected by Precision@50; shipped an editor playbook and a public research page with honest limitations.
