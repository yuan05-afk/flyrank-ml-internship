"""Deduplicate/fix Your-turn cells in notebooks 01 and 02. Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path


def md_src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    if not lines:
        return []
    return [l + "\n" for l in lines[:-1]] + [lines[-1] + "\n"]


def code_src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines]


def fix_nb01() -> None:
    p1 = Path("notebooks/01_first_look_and_discovery.ipynb")
    nb1 = json.loads(p1.read_text(encoding="utf-8"))

    strong_md = """### My discovery (Yuan)

**Discovery A (volume vs impressions):** `corr(search_volume, impressions_90d) ≈ 0.001`.
High keyword volume barely tracks impressions on this 30k starter slice.

**Discovery B (CTR at same position tier):** Holding `avg_position` in bands and
requiring `n ≥ 50`, **comparison articles** have the worst mean CTR in the
`4-10`, `11-20`, and `20+` tiers; **feedly articles** lead. In `1-3`, keyword
articles lag feedly.

**Decision-support takeaway:** Refresh priority should weight engagement + trend
(and content type under the same rank band), not search volume alone.

**Careful words:** Observed association in the starter snapshot, not proof that
refreshing comparison pages causes CTR recovery.
"""

    # Find your-turn prompt and the discovery code cell (has "# Your discovery here")
    your_turn_i = None
    discovery_code = None
    notes_cell = None
    save_cell = None
    for i, c in enumerate(nb1["cells"]):
        src = "".join(c.get("source", []))
        if "## 3." in src[:20] and "Your turn" in src:
            your_turn_i = i
        if src.strip().startswith("# Your discovery here"):
            discovery_code = c
        if "Yuan's discovery notes" in src or "Yuan's discovery notes (executed)" in src:
            notes_cell = c
        if "Save your work" in src:
            save_cell = c

    if your_turn_i is None or discovery_code is None:
        raise SystemExit(f"nb01 structure unexpected: your_turn={your_turn_i} code={discovery_code is not None}")

    head = nb1["cells"][: your_turn_i + 1]
    new_cells = head + [
        {
            "cell_type": "markdown",
            "id": "yuan-discovery-md",
            "metadata": {},
            "source": md_src(strong_md),
        },
        {
            "cell_type": "code",
            "id": "yuan-discovery-code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": discovery_code["source"],
        },
    ]
    if notes_cell is not None:
        new_cells.append(
            {
                "cell_type": "markdown",
                "id": "yuan-discovery-notes",
                "metadata": {},
                "source": notes_cell["source"],
            }
        )
    if save_cell is not None:
        new_cells.append(
            {
                "cell_type": "markdown",
                "id": "save-work",
                "metadata": {},
                "source": save_cell["source"],
            }
        )

    nb1["cells"] = new_cells
    p1.write_text(json.dumps(nb1, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"nb01 cells now {len(nb1['cells'])}")


def fix_nb02() -> None:
    p2 = Path("notebooks/02_your_first_readable_model.ipynb")
    nb2 = json.loads(p2.read_text(encoding="utf-8"))

    your_turn_code = r'''# Your turn — depth / feature try + honest client-holdout check
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd

# Build from df (not the earlier X frame) so we can try engagement_rate, etc.
# Never include trend_direction / trend_pct — those leak the declining label.
_candidates = [
    "impressions_90d", "ctr", "avg_position",
    "engagement_rate", "scroll_rate",
    "word_count", "content_age_days", "days_since_last_update",
]
feat_cols = [c for c in _candidates if c in df.columns]
X_use = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
y_use = pd.Series(df["is_declining_label"].to_numpy().astype(int)).reset_index(drop=True)
groups = pd.Series(df["client_id"].to_numpy()).reset_index(drop=True)
print("Using features:", feat_cols)

# In-sample depth-3 tree (teaching contrast only)
tree_in = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
tree_in.fit(X_use, y_use)
print(f"In-sample Precision@all (depth-3): {precision_score(y_use, tree_in.predict(X_use)):.3f}")
print(export_text(tree_in, feature_names=feat_cols))

# Client-holdout: train on ~80% of clients, score on unseen clients
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X_use, y_use, groups))
tree_ho = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
tree_ho.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
proba = tree_ho.predict_proba(X_use.iloc[te_idx])[:, 1]
y_te = y_use.iloc[te_idx]
print(f"Client-holdout Precision@all (depth-3): {precision_score(y_te, tree_ho.predict(X_use.iloc[te_idx])):.3f}")
print(f"Holdout clients={groups.iloc[te_idx].nunique()} pages={len(te_idx)}")
order = np.argsort(-proba)
for k in (20, 50):
    print(f"Holdout Precision@{k}: {y_te.iloc[order[:k]].mean():.3f}")
print(
    "Learning: if holdout Precision@K is below the hand rule or collapses vs in-sample, "
    "the tree was memorizing client mix; keep client-aware splits in the real pipeline."
)
'''

    notes = """### Yuan's Your-turn notes (executed)

**What I tried.** Depth-3 balanced tree on non-leaky features from `df`
(`impressions_90d`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`,
`word_count`, `content_age_days`, `days_since_last_update`). Compared in-sample
precision to a **client-holdout** split (`GroupShuffleSplit` on `client_id`,
test_size=0.2, seed=42), then reported Precision@20 / @50 on held-out clients.

**Learning.** In-sample scores flatter the model. Client-holdout is the honest bar
for a multi-tenant SEO warehouse: a client's pages must not appear in both train
and test. If Precision@K collapses on unseen clients, the model is not ready for
an editor-facing refresh queue.

**Product takeaway.** Keep the hand rule as the readable baseline; promote the
tree only when holdout Precision@K beats it without leaking trend labels.
"""

    cells = nb2["cells"]
    your_turn_i = None
    save_cell = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("## 4.") and "Your turn" in src:
            your_turn_i = i
        if "Save your work" in src:
            save_cell = c

    if your_turn_i is None:
        raise SystemExit("nb02: could not find Your turn section")

    new_cells = list(cells[: your_turn_i + 1]) + [
        {
            "cell_type": "markdown",
            "id": "yuan-yourturn-notes",
            "metadata": {},
            "source": md_src(notes),
        },
        {
            "cell_type": "code",
            "id": "yuan-yourturn-code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_src(your_turn_code),
        },
    ]
    if save_cell is not None:
        new_cells.append(
            {
                "cell_type": "markdown",
                "id": "save-work-02",
                "metadata": {},
                "source": save_cell["source"],
            }
        )

    nb2["cells"] = new_cells
    p2.write_text(json.dumps(nb2, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"nb02 cells now {len(nb2['cells'])}")


if __name__ == "__main__":
    fix_nb01()
    fix_nb02()
    print("done")
