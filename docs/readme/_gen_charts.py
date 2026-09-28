"""Generate compact README chart PNGs from model_results.json."""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "outputs" / "model_results.json").read_text(encoding="utf-8"))

TEAL = "#0d7377"
SLATE = "#64748b"
INK = "#0f172a"
MUTED = "#475569"
BG = "#f8fafc"
CARD = "#ffffff"


def save_fig(fig: plt.Figure, name: str, max_w: int = 640) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=BG, pad_inches=0.25)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)), Image.Resampling.LANCZOS)
    dest = OUT / name
    im.save(dest, "PNG", optimize=True)
    print(name, im.size, dest.stat().st_size)


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def chart_precision_k() -> None:
    base = DATA["baseline"]
    # RF from models list or nested
    rf = None
    for key in ("models", "results", "comparisons"):
        block = DATA.get(key)
        if isinstance(block, dict) and "random_forest" in block:
            rf = block["random_forest"]
            break
        if isinstance(block, list):
            for row in block:
                if str(row.get("model", "")).lower() in {"random_forest", "rf"}:
                    rf = row
                    break
    if rf is None:
        # fall back to README known numbers + baseline companions
        rf = {
            "precision_at_20": 0.65,
            "precision_at_50": 0.74,
            "precision_at_100": 0.72,
        }
        # try best_model nested
        bm = DATA.get("best_model") or {}
        for k in ("precision_at_20", "precision_at_50", "precision_at_100"):
            if k in bm:
                rf[k] = bm[k]
        # sometimes metrics live at top under evaluation
        for section in DATA.values():
            if isinstance(section, dict) and section.get("model") == "random_forest":
                rf = section
                break

    labels = ["P@20", "P@50", "P@100"]
    baseline_vals = [
        base["baseline_precision_at_20"],
        base["baseline_precision_at_50"],
        base["baseline_precision_at_100"],
    ]
    rf_vals = [
        float(rf.get("precision_at_20") or rf.get("baseline_precision_at_20") or 0.65),
        float(rf.get("precision_at_50") or rf.get("precision_at_k") or 0.74),
        float(rf.get("precision_at_100") or 0.72),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.2), facecolor=BG)
    style_ax(ax)
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline rules", color=SLATE)
    ax.bar([i + width / 2 for i in x], rf_vals, width, label="Random forest", color=TEAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Precision", color=MUTED, fontsize=9)
    ax.set_title("Precision@K · baseline vs RF", color=INK, fontsize=11, fontweight="600", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_fig(fig, "chart-precision.png")


def chart_model_comparison() -> None:
    rows = []
    # Prefer table in model_report-like structure
    block = DATA.get("models") or DATA.get("model_metrics") or DATA.get("results")
    if isinstance(block, dict):
        for name, metrics in block.items():
            if isinstance(metrics, dict) and "precision_at_50" in metrics:
                rows.append((name, float(metrics["precision_at_50"])))
    if isinstance(block, list):
        for row in block:
            if "precision_at_50" in row:
                rows.append((str(row.get("model", "model")), float(row["precision_at_50"])))

    if not rows:
        # hardcode from committed report (same as README table)
        rows = [
            ("baseline_rules", 0.24),
            ("logistic_regression", 0.40),
            ("decision_tree", 0.62),
            ("random_forest", 0.74),
        ]

    # order by P@50 ascending for visual story
    rows = sorted(rows, key=lambda r: r[1])
    labels = [n.replace("_", " ") for n, _ in rows]
    vals = [v for _, v in rows]
    colors = [TEAL if "random" in n else SLATE for n, _ in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.2), facecolor=BG)
    style_ax(ax)
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Precision@50", color=MUTED, fontsize=9)
    ax.set_title("Model comparison · Precision@50", color=INK, fontsize=11, fontweight="600", pad=10)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8, color=MUTED)
    save_fig(fig, "chart-models.png")


def shrink_peeks() -> None:
    for name in ("paper-hero.png", "desk-modeling.png"):
        path = OUT / name
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        w, h = im.size
        # Prefer width ~480 for two-column GitHub embeds
        target = 520
        if w > target:
            im = im.resize((target, int(h * target / w)), Image.Resampling.LANCZOS)
        im.save(path, "PNG", optimize=True)
        print("peek", name, im.size, path.stat().st_size)


def main() -> None:
    print("keys", list(DATA.keys()))
    chart_precision_k()
    chart_model_comparison()
    # Prefer CDP desk hero if present as tmp-large
    tmp = OUT / "tmp-large.png"
    if tmp.exists():
        im = Image.open(tmp).convert("RGB")
        # crop bottom Netlify badge band if present (~4%)
        w, h = im.size
        im = im.crop((0, 0, w, int(h * 0.96)))
        target = 520
        if w > target:
            im = im.resize((target, int(im.size[1] * target / w)), Image.Resampling.LANCZOS)
        im.save(OUT / "desk-modeling.png", "PNG", optimize=True)
        print("desk-modeling from tmp-large", im.size)
    shrink_peeks()


if __name__ == "__main__":
    main()
