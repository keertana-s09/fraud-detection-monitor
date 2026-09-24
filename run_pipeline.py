#!/usr/bin/env python3
"""
End-to-end fraud detection pipeline with drift and monitoring.

This script demonstrates the full lifecycle:
  1. Generate realistic transaction data with evolving fraud patterns
  2. Train an imbalance-aware ensemble model
  3. Deploy to simulated production with concept drift
  4. Monitor for silent degradation
  5. Detect when retraining is needed
  6. Visualize everything

Run: python run_pipeline.py
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from data_simulator import TransactionSimulator, DriftConfig, add_engineered_features
from model import (
    FraudDetector, temporal_train_test_split, evaluate_model,
    train_and_evaluate, FEATURE_COLS, LABEL_COL, TIME_COL
)
from drift_detector import DriftMonitoringSuite
from monitor import ProductionMonitor

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_data_exploration(df: pd.DataFrame):
    """Visualize the dataset and its drift properties."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Transaction Data Exploration & Drift Visualization", fontsize=14, y=1.02)

    # 1. Fraud rate over time
    ax = axes[0, 0]
    weekly = df.set_index("timestamp").resample("W")["is_fraud"].agg(["mean", "sum"])
    ax.plot(weekly.index, weekly["mean"] * 100, color="crimson", linewidth=1.5)
    ax.fill_between(weekly.index, 0, weekly["mean"] * 100, alpha=0.2, color="crimson")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate Over Time (Weekly)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, alpha=0.3)

    # 2. Fraud type composition over time
    ax = axes[0, 1]
    fraud_only = df[df["is_fraud"] == 1].copy()
    fraud_only["month"] = fraud_only["timestamp"].dt.to_period("M")
    type_counts = fraud_only.groupby(["month", "fraud_type"]).size().unstack(fill_value=0)
    type_pcts = type_counts.div(type_counts.sum(axis=1), axis=0) * 100
    type_pcts.index = type_pcts.index.to_timestamp()
    colors = {"cnp": "#e74c3c", "ato": "#3498db", "bust_out": "#2ecc71", "micro_test": "#9b59b6"}
    type_pcts.plot.area(ax=ax, alpha=0.7, color=[colors.get(c, "#999") for c in type_pcts.columns])
    ax.set_ylabel("Fraud Mix (%)")
    ax.set_title("Fraud Type Composition Drift")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Transaction amount distribution: legit vs fraud
    ax = axes[0, 2]
    ax.hist(df[df["is_fraud"] == 0]["amount"].clip(upper=500), bins=80,
            alpha=0.6, label="Legitimate", color="steelblue", density=True)
    ax.hist(df[df["is_fraud"] == 1]["amount"].clip(upper=500), bins=80,
            alpha=0.6, label="Fraud", color="crimson", density=True)
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Density")
    ax.set_title("Amount Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Fraud by hour of day
    ax = axes[1, 0]
    legit_hours = df[df["is_fraud"] == 0]["hour_of_day"].value_counts().sort_index()
    fraud_hours = df[df["is_fraud"] == 1]["hour_of_day"].value_counts().sort_index()
    fraud_rate_by_hour = fraud_hours / (legit_hours + fraud_hours) * 100
    ax.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color="crimson", alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour")
    ax.grid(True, alpha=0.3)

    # 5. Feature correlations with fraud
    ax = axes[1, 1]
    numeric_cols = ["amount", "is_online", "distance_from_home_km", "is_international",
                    "num_transactions_last_1h", "is_new_merchant", "is_night"]
    correlations = df[numeric_cols + ["is_fraud"]].corr()["is_fraud"].drop("is_fraud").sort_values()
    correlations.plot.barh(ax=ax, color=["crimson" if v > 0 else "steelblue" for v in correlations])
    ax.set_title("Feature Correlation with Fraud")
    ax.grid(True, alpha=0.3)

    # 6. Class imbalance visualization
    ax = axes[1, 2]
    counts = df["is_fraud"].value_counts()
    bars = ax.bar(["Legitimate", "Fraud"], counts.values,
                  color=["steelblue", "crimson"], alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title(f"Class Distribution ({counts[1]/len(df):.2%} fraud)")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + len(df)*0.005,
                f"{count:,}", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_data_exploration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/01_data_exploration.png")


def plot_model_performance(test_df: pd.DataFrame, detector: FraudDetector):
    """Visualize model performance on test data."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    scores = detector.predict_proba(test_df)
    labels = test_df[LABEL_COL].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Performance Analysis", fontsize=14)

    # 1. Precision-Recall curve
    ax = axes[0]
    prec, rec, thresh = precision_recall_curve(labels, scores)
    ax.plot(rec, prec, color="crimson", linewidth=2)
    ax.fill_between(rec, prec, alpha=0.1, color="crimson")
    ax.axhline(y=labels.mean(), color="gray", linestyle="--", label=f"Random: {labels.mean():.4f}")
    ax.axvline(x=detector.threshold, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall (Fraud Catch Rate)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AUPRC={average_precision_score(labels, scores):.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Score distribution
    ax = axes[1]
    ax.hist(scores[labels == 0], bins=100, alpha=0.6, label="Legitimate",
            color="steelblue", density=True, log=True)
    ax.hist(scores[labels == 1], bins=100, alpha=0.6, label="Fraud",
            color="crimson", density=True, log=True)
    ax.axvline(x=detector.threshold, color="green", linestyle="--",
               label=f"Threshold: {detector.threshold:.3f}", linewidth=2)
    ax.set_xlabel("Fraud Score")
    ax.set_ylabel("Density (log)")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Feature importance
    ax = axes[2]
    importance = detector.get_feature_importance().head(12)
    bars = ax.barh(importance["feature"], importance["avg_importance"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Top Feature Importance (Ensemble)")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/02_model_performance.png")


def plot_production_monitoring(history_df: pd.DataFrame, alerts_df: pd.DataFrame,
                                baseline_auprc: float, baseline_recall: float):
    """Visualize production monitoring results."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Production Monitoring Dashboard", fontsize=14, y=1.02)

    history = history_df[history_df["n_fraud"] > 0].copy()

    # 1. AUPRC over time with degradation threshold
    ax = axes[0, 0]
    ax.plot(history["window_start"], history["auprc"], "o-",
            color="crimson", linewidth=2, markersize=4, label="AUPRC")
    ax.axhline(y=baseline_auprc, color="green", linestyle="--",
               label=f"Baseline: {baseline_auprc:.3f}", alpha=0.8)
    ax.axhline(y=baseline_auprc * 0.85, color="orange", linestyle=":",
               label="Warning threshold", alpha=0.8)
    # Shade degradation zone
    ax.fill_between(history["window_start"], 0, baseline_auprc * 0.85,
                     alpha=0.05, color="red")
    ax.set_ylabel("AUPRC")
    ax.set_title("Model Performance Over Time (AUPRC)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 2. Precision and Recall over time
    ax = axes[0, 1]
    ax.plot(history["window_start"], history["precision"], "s-",
            color="steelblue", linewidth=1.5, markersize=4, label="Precision")
    ax.plot(history["window_start"], history["recall"], "^-",
            color="crimson", linewidth=1.5, markersize=4, label="Recall")
    ax.axhline(y=baseline_recall, color="crimson", linestyle="--", alpha=0.3)
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall Over Time")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)

    # 3. Score distribution shift over time
    ax = axes[1, 0]
    ax.plot(history["window_start"], history["mean_score"], "o-",
            color="purple", linewidth=1.5, markersize=4, label="Mean Score")
    ax.fill_between(history["window_start"],
                     history["mean_score"] - history["score_std"],
                     history["mean_score"] + history["score_std"],
                     alpha=0.2, color="purple")
    ax.plot(history["window_start"], history["score_p95"], "x--",
            color="orange", linewidth=1, label="95th Percentile")
    ax.set_ylabel("Score")
    ax.set_title("Score Distribution Over Time")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)

    # 4. Alert timeline
    ax = axes[1, 1]
    if len(alerts_df) > 0:
        alert_types = alerts_df["type"].unique()
        colors_map = {"psi": "#e74c3c", "ks": "#3498db", "adwin": "#2ecc71",
                      "label_drift": "#f39c12", "degradation": "#8e44ad"}
        for atype in alert_types:
            subset = alerts_df[alerts_df["type"] == atype]
            marker = "v" if subset["severity"].iloc[0] == "critical" else "^"
            ax.scatter(subset["timestamp"], [atype] * len(subset),
                      c=colors_map.get(atype, "#999"), s=80, marker=marker,
                      label=f"{atype} ({len(subset)})", alpha=0.7, edgecolors="black")
        ax.set_title(f"Alert Timeline ({len(alerts_df)} total alerts)")
        ax.legend(fontsize=9, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No alerts generated", ha="center", va="center",
                fontsize=14, color="gray", transform=ax.transAxes)
        ax.set_title("Alert Timeline")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_production_monitoring.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03_production_monitoring.png")


def plot_drift_analysis(df: pd.DataFrame, detector: FraudDetector):
    """
    Show how feature distributions shift between early and late data,
    demonstrating why the model degrades.
    """
    mid = len(df) // 2
    early = df.iloc[:mid]
    late = df.iloc[mid:]

    features_to_show = ["amount", "hour_of_day", "distance_from_home_km",
                        "num_transactions_last_1h", "is_new_merchant", "is_online"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature Distribution Drift: Early vs Late Period (Fraud Only)", fontsize=14)

    early_fraud = early[early["is_fraud"] == 1]
    late_fraud = late[late["is_fraud"] == 1]

    for idx, feat in enumerate(features_to_show):
        ax = axes[idx // 3, idx % 3]

        if feat in ("is_new_merchant", "is_online"):
            # Binary features — show as bar chart
            early_rate = early_fraud[feat].mean()
            late_rate = late_fraud[feat].mean()
            bars = ax.bar(["Early\nFraud", "Late\nFraud"],
                         [early_rate, late_rate],
                         color=["steelblue", "crimson"], alpha=0.7)
            ax.set_ylabel("Rate")
            for bar, val in zip(bars, [early_rate, late_rate]):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f"{val:.2%}", ha="center", fontsize=10)
        else:
            ax.hist(early_fraud[feat].clip(upper=early_fraud[feat].quantile(0.99)),
                    bins=50, alpha=0.5, label="Early", color="steelblue", density=True)
            ax.hist(late_fraud[feat].clip(upper=late_fraud[feat].quantile(0.99)),
                    bins=50, alpha=0.5, label="Late", color="crimson", density=True)
            ax.legend()
            ax.set_ylabel("Density")

        ax.set_title(f"{feat}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_drift_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/04_drift_analysis.png")


from sklearn.metrics import average_precision_score


def main():
    print("=" * 80)
    print("  FRAUD DETECTION PIPELINE WITH CONCEPT DRIFT & MONITORING")
    print("=" * 80)

    # ── Step 1: Generate data with drift ─────────────────────────────────────
    print("\n[1/5] Generating transaction data with concept drift...")
    sim = TransactionSimulator(seed=42, drift_config=DriftConfig())
    df = sim.generate(n_transactions=150_000, n_customers=10_000, n_days=365)
    df = add_engineered_features(df)

    plot_data_exploration(df)

    # ── Step 2: Train model on early data ────────────────────────────────────
    print("\n[2/5] Training fraud detection ensemble...")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    # 60% train, 10% validation, 30% held out for production simulation
    train_df = df.iloc[:int(n * 0.6)]
    val_df = df.iloc[int(n * 0.6):int(n * 0.7)]
    prod_df = df.iloc[int(n * 0.7):]  # This simulates "future" production data

    print(f"\nTrain period: {train_df[TIME_COL].min().date()} to {train_df[TIME_COL].max().date()}")
    print(f"Val period:   {val_df[TIME_COL].min().date()} to {val_df[TIME_COL].max().date()}")
    print(f"Prod period:  {prod_df[TIME_COL].min().date()} to {prod_df[TIME_COL].max().date()}")

    detector = FraudDetector(use_smote=False)
    detector.fit(train_df, val_df)

    # ── Step 3: Evaluate on validation data ──────────────────────────────────
    print("\n[3/5] Evaluating model...")
    val_scores = detector.predict_proba(val_df)
    val_labels = val_df[LABEL_COL].values
    val_metrics = evaluate_model(val_labels, val_scores, detector.threshold,
                                 label="Validation Set (Temporal)")

    plot_model_performance(val_df, detector)

    # ── Step 4: Simulate production with monitoring ──────────────────────────
    print("\n[4/5] Simulating production deployment with monitoring...")
    monitor = ProductionMonitor(
        model=detector,
        feature_cols=FEATURE_COLS,
        window_size_days=7,
        degradation_threshold=0.15,
    )
    monitor.set_baseline(train_df, val_df)

    history_df = monitor.simulate_production(prod_df)

    # ── Step 5: Analyze results and generate recommendations ─────────────────
    print("\n[5/5] Analyzing monitoring results...")

    # Get alerts
    alerts_df = monitor.drift_suite.get_alert_summary()
    if len(alerts_df) > 0:
        print(f"\nAlert Summary:")
        print(f"  Total alerts: {len(alerts_df)}")
        for atype in alerts_df["type"].unique():
            count = len(alerts_df[alerts_df["type"] == atype])
            print(f"    {atype}: {count}")

    # Get retraining recommendation
    recommendation = monitor.get_retraining_recommendation()
    print(f"\nRetraining Recommendation:")
    print(f"  Should retrain: {recommendation['should_retrain']}")
    print(f"  Urgency: {recommendation['urgency']}")
    if recommendation["reasons"]:
        print(f"  Reasons:")
        for reason in recommendation["reasons"]:
            print(f"    - {reason}")
    print(f"  Strategy: {recommendation['suggested_strategy']}")

    # Plot monitoring dashboard
    plot_production_monitoring(
        history_df, alerts_df,
        monitor.baseline_auprc, monitor.baseline_recall
    )

    # Plot drift analysis
    plot_drift_analysis(df, detector)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Outputs saved to: {OUTPUT_DIR.absolute()}")
    print(f"  - 01_data_exploration.png    : Dataset visualization & class imbalance")
    print(f"  - 02_model_performance.png   : PR curve, scores, feature importance")
    print(f"  - 03_production_monitoring.png: Performance degradation over time")
    print(f"  - 04_drift_analysis.png      : Feature distribution shifts")
    print(f"\n  Key findings:")
    print(f"    Model trained on early data (baseline AUPRC: {monitor.baseline_auprc:.3f})")

    if recommendation["recent_auprc"]:
        drop = ((monitor.baseline_auprc - recommendation["recent_auprc"])
                / monitor.baseline_auprc * 100)
        print(f"    Recent AUPRC: {recommendation['recent_auprc']:.3f} "
              f"({drop:+.1f}% from baseline)")

    print(f"    Drift alerts detected: {len(alerts_df) if len(alerts_df) else 0}")
    print(f"    Retraining urgency: {recommendation['urgency']}")
    print()


if __name__ == "__main__":
    main()
