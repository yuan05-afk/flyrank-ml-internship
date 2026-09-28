# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Yuan Andrei C. Mariano
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/yuan05-afk/flyrank-ml-internship
- **Date:** 2026-09-29

## 0. Abstract

Editors need a short queue of which pages to refresh first. Using FlyRank’s 30,000-row anonymized starter snapshot (32 clients; trailing-90-day metrics), this project ranks content items for review with an observed decline label, a transparent rule baseline, and client-holdout validation. A random forest selected by Precision@50 reaches 0.74 on holdout versus 0.24 for the baseline (base rate ≈ 0.54). The shipped output is a ranked review queue with scores, actions, and reason codes for human triage, not an automatic publish decision.

## 1. Problem framing

**Decision:** Which pages should an editor refresh-review first?
**Unit:** One content item (page).
**Output:** Ranked queue with score, action, reason codes.
**Action:** Editor prioritizes rewrite / CTR review / expand.
**Cost of wrong call:** Wasted rewrite effort or missed declining visible pages.
**Why ML:** Multi-signal interactions are messy for one if-statement; ML must beat the transparent baseline on Precision@K.

## 2. Data safety

Used starter CSV only in this build. Excluded `trend_direction` / `trend_pct` as features; IDs are split keys only. No client names, URLs, or private queries. Warehouse access deferred (no HF token); documented in SETUP.md.

## 3. Baseline

Visibility × freshness risk × CTR-gap flag. Reason codes: `stale_visible_page`, `low_ctr_visible_page`, `general_refresh_review`. Client-holdout Precision@50 ≈ 0.24 (reference pipeline).

## 4. Model / analysis

LR / DT / RF on `ml_utils` feature lists. Target: `is_declining_label`. Best: random forest by Precision@50.

## 5. Evaluation

Split: client-holdout. RF Precision@50 = 0.74, ROC-AUC = 0.75, AP = 0.618 vs baseline 0.24 / 0.627 / 0.468. Base rate ≈ 0.542. Error pattern: high-impression stable false positives; low-volume miss risk.

## 6. Interpretation

Top features: days_with_impressions, log_impressions_90d, avg_position, content_age_days. Directional association with the decline label under holdout; not causal recovery proof.

## 7. Recommendation

Triage high-confidence refresh / CTR-review rows; human verify; monitor Precision@50 vs baseline after data refreshes. See `work/notebooks/w07_action_playbook.ipynb`.

## 8. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

Seed 42 in work notebooks. Receipts: `outputs/model_results.json`, `work/outputs/*.json`.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
