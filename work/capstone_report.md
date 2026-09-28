# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Yuan Andrei C. Mariano
- **Program:** Computer Science · Specialization in Data Science
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/yuan05-afk/flyrank-ml-internship
- **Paper (primary):** https://yuan-mariano-works.netlify.app/ml/paper/
- **Date:** 2026-09-29

## Title

Ranking Content for Refresh Review from Observable Search Signals

## 0. Abstract (5 sentences)

Which existing pages should an editor open for refresh review first when rewrite capacity is scarce? Using FlyRank’s 30,000-row anonymized starter snapshot (32 clients; trailing-90-day search and engagement metrics), this project frames the decision as a ranking task with an observed decline label, a transparent rule baseline, and client-holdout validation. Models are fit without titles, URLs, client names, or private queries, and without using trend fields as features. A random forest selected by Precision@50 reaches 0.74 on the holdout versus 0.24 for the baseline (base rate ≈ 0.54; ROC-AUC ≈ 0.75). The shipped output is a ranked review queue with scores, actions, and reason codes for human triage, not an automatic publish decision and not a claim about Google’s ranking algorithm.

## 1. Introduction / Problem

**Decision:** Which pages should an editor refresh-review first?
**Unit:** One content item (page).
**Output:** Ranked queue with score, action, reason codes.
**Action:** Editor prioritizes rewrite / CTR review / expand.
**Cost of wrong call:** Wasted rewrite effort or missed declining visible pages.
**Why ML:** Multi-signal interactions are messy for one if-statement; ML must beat the transparent baseline on Precision@K on the same split.

## 2. Data

| Field | Value |
|---|---|
| Release used | `data/raw/content_refresh_anonymized.csv` (starter) |
| Rows × cols | 30,000 × 44 |
| Clients | 32 |
| Window | Trailing-90-day aggregates on the snapshot |
| Declining-label rate | ≈ 0.542 (16,262 / 30,000) |

**Warehouse (deferred):** Hugging Face `FlyRank/internship-warehouse` (~79M daily rows) was not queried (no HF token in this build). SETUP.md documents the access path for future panel work. This report does not pretend warehouse data was used.

**Exclusions (public-safe):** No client names, domains, URLs, or private queries. `trend_direction` / `trend_pct` excluded as features (label definition). IDs are split keys only.

**Gotchas:** Rate columns are ×100 percentages. `avg_position = 0` means missing rank (~4.02%), not rank zero.

## 3. Methodology

**Assumptions:** Short weekly queues; Precision@K is the operational metric; observed decline proxy is acceptable when labeled honestly; client-holdout approximates unseen clients.

**Label:** `is_declining_label` when `trend_direction == down` in the snapshot.

**Features:** Lists in `scripts/ml_utils.py` (impressions/clicks/sessions transforms, freshness, CTR, position, content tiers). Leakage audit: forbidden trend fields empty in features (`work/outputs/w06_leakage_audit.json`).

**Baseline:** Visibility × freshness risk × CTR-gap flag with reason codes (`stale_visible_page`, `low_ctr_visible_page`, `general_refresh_review`).

**Models:** Logistic regression, decision tree, random forest.

**Validation:** Client-holdout (train 27,675 / test 2,325). Selection metric: Precision@50. Seed 42 in work notebooks. Reference pipeline: `python scripts/run_all.py`.

## 4. Results

Best model: **random_forest** by Precision@50.

| Model | Precision@50 | P@20 | P@100 | ROC-AUC | Avg precision |
|---|---:|---:|---:|---:|---:|
| baseline_rules | 0.24 | 0.15 | 0.36 | 0.627 | 0.468 |
| logistic_regression | 0.40 | 0.35 | 0.44 | 0.700 | 0.522 |
| decision_tree | 0.62 | 0.55 | 0.60 | 0.742 | 0.575 |
| **random_forest** | **0.74** | **0.65** | **0.72** | **0.750** | **0.618** |

**Top features:** `days_with_impressions`, `log_impressions_90d`, `avg_position`, `content_age_days`.

**Queue mix (reference):** high-confidence 3,602; monitor 13,083; refresh 8,188; refresh_and_review_ctr 6,654.

Charts: `outputs/charts/*.svg` and live paper figures.

## 5. Limitations

- Snapshot observed label, not a sealed future warehouse window.
- Starter sample; warehouse deferred without HF credentials.
- Measured / directional / decision-support under holdout — not causal recovery proof.
- No claim of predicting Google’s algorithm.
- Rate-scale and missingness gotchas can mislead naive feature engineering.

## 6. Ranked recommendations / action playbook

1. Open high-confidence `refresh` / `refresh_and_review_ctr` rows first.
2. Verify on-page context; apply no-go list (legal freezes, YMYL caution, no auto-publish).
3. Treat `monitor` as backlog, not rewrite fuel.
4. Retrain or freeze when holdout Precision@50 drifts toward the baseline across refreshes.

See `work/notebooks/w07_action_playbook.ipynb` and `work/outputs/playbook_*.json`.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

Notebooks: `work/notebooks/` (w01–w07 + `capstone.ipynb`) and starter `notebooks/01`, `02`.
Receipts: `outputs/model_results.json`, `work/outputs/*.json`.
Showcase desks: https://yuan-mariano-works.netlify.app/#ml

## 8. Acknowledgments and data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
