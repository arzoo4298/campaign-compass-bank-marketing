"""Reproducible campaign analysis for the UCI Bank Marketing data."""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
IMAGE_DIR = ROOT / "images"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_MEMBER = "bank-additional-full.csv"


def load_data(offline: bool = False) -> pd.DataFrame:
    """Load cached UCI CSV, downloading the official archive when needed."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "bank-additional-full.csv"
    if not csv_path.exists():
        if offline:
            raise FileNotFoundError(
                f"{csv_path} is missing. Run once with internet access and without --offline."
            )
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "CampaignCompass/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_archive:
            nested_archives = [name for name in outer_archive.namelist() if name.endswith("bank-additional.zip")]
            if len(nested_archives) != 1:
                raise ValueError(f"Expected one bank-additional.zip in the UCI archive; found {nested_archives}.")
            with zipfile.ZipFile(io.BytesIO(outer_archive.read(nested_archives[0]))) as archive:
                matches = [name for name in archive.namelist() if name.endswith(CSV_MEMBER)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {CSV_MEMBER} in the nested archive; found {matches}.")
                with archive.open(matches[0]) as source, csv_path.open("wb") as target:
                    target.write(source.read())
    return pd.read_csv(csv_path, sep=";", low_memory=False)


def make_encoder():
    """Support both sklearn's newer sparse_output and older sparse parameter."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def save_segment_rates(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in ("poutcome", "job", "contact", "month", "education", "marital"):
        grouped = data.groupby(column, dropna=False)["subscribed"].agg(
            customers="size", subscriptions="sum", conversion_rate="mean"
        ).reset_index().rename(columns={column: "segment"})
        grouped.insert(0, "dimension", column)
        frames.append(grouped)
    result = pd.concat(frames, ignore_index=True)
    result["conversion_rate_pct"] = (100 * result["conversion_rate"]).round(2)
    result.to_csv(REPORT_DIR / "segment_rates.csv", index=False)
    return result


def plot_previous_outcome(data: pd.DataFrame) -> None:
    stats = data.groupby("poutcome", observed=True)["subscribed"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    bars = ax.bar(stats.index.astype(str), stats.values * 100, color="#2563eb")
    ax.set(title="Subscription rate varies with prior campaign outcome",
           ylabel="Subscribed (%)", xlabel="Outcome of previous campaign")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, stats.values * 100):
        ax.text(bar.get_x() + bar.get_width()/2, value + .12, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "conversion_by_previous_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deciles(y_true: pd.Series, scores: np.ndarray) -> float:
    scored = pd.DataFrame({"actual": np.asarray(y_true), "score": scores})
    scored["decile"] = pd.qcut(scored["score"].rank(method="first"), 10, labels=False) + 1
    rates = scored.groupby("decile", observed=True)["actual"].mean().sort_index(ascending=False)
    counts = scored.groupby("decile", observed=True).size().reindex(rates.index)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.bar([f"D{i}" for i in rates.index], rates.values * 100, color="#0f766e")
    ax.axhline(scored["actual"].mean() * 100, color="#dc2626", linestyle="--", label="Test-set average")
    ax.set(title="Observed subscription rate by model score decile",
           ylabel="Subscribed (%)", xlabel="Highest score first")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "decile_lift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    top_rate = float(rates.iloc[0])
    overall = float(scored["actual"].mean())
    return top_rate / overall if overall else float("nan")


def main(offline: bool = False) -> None:
    for path in (REPORT_DIR, IMAGE_DIR):
        path.mkdir(exist_ok=True)

    data = load_data(offline)
    required = {"y", "duration", "campaign", "poutcome", "job", "contact", "month"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    data["subscribed"] = data["y"].map({"yes": 1, "no": 0})
    if data["subscribed"].isna().any():
        raise ValueError("Unexpected values found in target column y.")
    data["subscribed"] = data["subscribed"].astype("int8")

    # These chronological rows are ordered by date in the source. Reserve the newest
    # fifth as an out-of-time holdout to better reflect deployment on future campaigns.
    split_at = int(len(data) * .8)
    train, test = data.iloc[:split_at].copy(), data.iloc[split_at:].copy()
    excluded = {"y", "subscribed", "duration", "campaign"}
    feature_columns = [column for column in data.columns if column not in excluded]
    X_train, X_test = train[feature_columns], test[feature_columns]
    y_train, y_test = train["subscribed"], test["subscribed"]
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in feature_columns if column not in numeric]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", make_encoder())]), categorical),
    ])
    model = Pipeline([("prepare", preprocessor),
                      ("ranker", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    model.fit(X_train, y_train)
    positive_col = list(model.named_steps["ranker"].classes_).index(1)
    scores = model.predict_proba(X_test)[:, positive_col]
    overall = float(data["subscribed"].mean())
    train_rate = float(y_train.mean())
    test_rate = float(y_test.mean())
    top_decile_lift = plot_deciles(y_test, scores)

    rates = save_segment_rates(data)
    plot_previous_outcome(data)
    metrics = {
        "rows": int(len(data)), "features": feature_columns,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_period": "first 80% of source rows (source is date-ordered)",
        "test_period": "most recent 20% of source rows (out-of-time holdout)",
        "overall_conversion_rate": overall, "train_conversion_rate": train_rate,
        "test_conversion_rate": test_rate,
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "top_decile_lift": top_decile_lift,
        "sklearn_version": sklearn.__version__,
    }
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best = rates[rates["dimension"].eq("poutcome")].sort_values("conversion_rate", ascending=False)
    lines = [
        "# Campaign analysis: decision memo", "",
        "## Recommendation", "",
        "Use this analysis to design a controlled targeting pilot. Do not treat the model ranking as proof that extra calls cause more subscriptions.", "",
        "## Dataset and quality", "",
        f"- {len(data):,} records; {int(data['subscribed'].sum()):,} subscriptions; overall historical subscription rate **{overall:.1%}**.",
        f"- The chronological holdout contains {len(test):,} later records with a **{test_rate:.1%}** subscription rate, compared with **{train_rate:.1%}** in training.",
        "- The response rate shifts sharply across this time split. Treat the holdout as a drift stress test; its lift estimate may not transfer to a current campaign.",
        "- The source includes contact duration, which is observed only after a call. It is excluded. Current-campaign call count is excluded because it changes during a campaign.",
        "- This is historical data from 2008–2010. Channel, customer mix, prices, and response behavior may have changed.", "",
        "## Model readout", "",
        f"- Pre-contact feature baseline: ROC-AUC **{metrics['roc_auc']:.3f}**, average precision **{metrics['average_precision']:.3f}** on the most recent 20% of rows.",
        f"- Highest-scored decile conversion rate is **{float(np.asarray(y_test)[np.argsort(scores)[-max(1, len(scores)//10):]].mean()):.1%}**; {top_decile_lift:.2f}× the holdout average.",
        "- AUC measures ranking across thresholds; average precision is useful with an imbalanced outcome. Neither measures campaign profit or causal lift.", "",
        "## Segment signal", "",
        "Historical subscription rates differ by previous campaign outcome:", "",
    ]
    for _, row in best.iterrows():
        lines.append(f"- `{row['segment']}`: {row['conversion_rate_pct']:.2f}% ({int(row['customers']):,} customers)")
    lines += ["", "Segment differences are descriptive, and small groups can have noisy rates. Review counts in `segment_rates.csv` before acting.", "",
              "## Pilot design", "",
              "1. Define an eligible population and contact-frequency cap with campaign owners.",
              "2. Randomly split eligible customers: model-ranked outreach versus business-as-usual outreach (or an uncontacted holdout if ethically and operationally suitable).",
              "3. Compare incremental subscriptions per 1,000 eligible customers and net value per contact; pre-specify guardrails and subgroup checks.",
              "4. Expand only if the pilot shows incremental value, calibrated scores, and acceptable contact and fairness outcomes.", "",
              "## Limitations", "",
              "This dataset contains no contact costs, deposit value, eligibility policy, or randomized assignment. It cannot support an ROI estimate or prove that contacting a high-scoring customer causes a subscription. The chronological holdout is more realistic than a random split, but it is still from the same historical source and era.", ""]
    (REPORT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis complete. Rows={len(data):,}; conversion={overall:.1%}; ROC-AUC={metrics['roc_auc']:.3f}; AP={metrics['average_precision']:.3f}")
    print(f"Wrote outputs under {REPORT_DIR} and {IMAGE_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require the dataset CSV to already exist in data/.")
    main(offline=parser.parse_args().offline)







