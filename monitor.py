"""
Production monitoring layer for fraud detection models.

This module simulates what happens after a model is deployed: transactions
stream in, the model scores them, and we need to detect when performance
is silently degrading — before fraud losses spike.

Key monitoring capabilities:
  - Sliding window performance tracking (precision, recall, AUPRC over time)
  - Score distribution monitoring (are scores shifting?)
  - Degradation detection with configurable thresholds
  - Automated retraining trigger recommendations
  - Alert aggregation and deduplication

In a real deployment, this would integrate with:
  - Prometheus/Grafana for metrics
  - PagerDuty/Opsgenie for alerting
  - MLflow/Weights & Biases for experiment tracking
  - Airflow/Prefect for retraining orchestration
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from dataclasses import dataclass, field
from typing import Optional
from drift_detector import DriftMonitoringSuite, DriftAlert


@dataclass
class PerformanceSnapshot:
    """Performance metrics for a single monitoring window."""
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_transactions: int
    n_fraud: int
    fraud_rate: float
    precision: float
    recall: float
    f1: float
    auprc: float
    roc_auc: float
    alert_rate: float
    mean_score: float
    score_std: float
    score_p95: float


class ProductionMonitor:
    """
    Simulates production monitoring of a deployed fraud model.

    Processes transactions in time-ordered windows, tracking how model
    performance evolves. Detects both sudden drops (data pipeline issues)
    and gradual degradation (concept drift).
    """

    def __init__(self, model, feature_cols: list,
                 window_size_days: int = 7,
                 degradation_threshold: float = 0.15):
        """
        Args:
            model: Trained FraudDetector instance
            feature_cols: Features the model uses
            window_size_days: Size of each monitoring window
            degradation_threshold: Relative drop in AUPRC that triggers alert
                                   (0.15 = 15% drop from baseline)
        """
        self.model = model
        self.feature_cols = feature_cols
        self.window_size_days = window_size_days
        self.degradation_threshold = degradation_threshold

        # Initialize drift monitoring suite
        self.drift_suite = DriftMonitoringSuite(feature_cols)

        # Tracking state
        self.performance_history = []
        self.baseline_auprc = None
        self.baseline_precision = None
        self.baseline_recall = None
        self.degradation_alerts = []

    def set_baseline(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Establish baseline metrics from validation data.
        These are the performance numbers we expect in production.
        """
        train_scores = self.model.predict_proba(train_df)
        val_scores = self.model.predict_proba(val_df)
        val_preds = self.model.predict(val_df)
        y_val = val_df["is_fraud"].values

        self.baseline_auprc = average_precision_score(y_val, val_scores)
        self.baseline_precision = precision_score(y_val, val_preds, zero_division=0)
        self.baseline_recall = recall_score(y_val, val_preds, zero_division=0)

        # Set reference distributions for drift detection
        self.drift_suite.set_reference(train_df, train_scores)

        print(f"Baseline established:")
        print(f"  AUPRC:     {self.baseline_auprc:.4f}")
        print(f"  Precision: {self.baseline_precision:.4f}")
        print(f"  Recall:    {self.baseline_recall:.4f}")

    def process_window(self, window_df: pd.DataFrame) -> tuple:
        """
        Process a single monitoring window.

        Returns:
            (PerformanceSnapshot, list of DriftAlerts)
        """
        if len(window_df) == 0:
            return None, []

        scores = self.model.predict_proba(window_df)
        preds = self.model.predict(window_df)
        labels = window_df["is_fraud"].values
        timestamp = window_df["timestamp"].iloc[-1]

        # Calculate window metrics
        n_fraud = labels.sum()
        if n_fraud == 0:
            # Can't compute fraud-dependent metrics without positive examples
            snapshot = PerformanceSnapshot(
                window_start=window_df["timestamp"].min(),
                window_end=window_df["timestamp"].max(),
                n_transactions=len(window_df),
                n_fraud=0,
                fraud_rate=0.0,
                precision=0.0, recall=0.0, f1=0.0,
                auprc=0.0, roc_auc=0.5,
                alert_rate=preds.mean(),
                mean_score=scores.mean(),
                score_std=scores.std(),
                score_p95=np.percentile(scores, 95),
            )
        else:
            snapshot = PerformanceSnapshot(
                window_start=window_df["timestamp"].min(),
                window_end=window_df["timestamp"].max(),
                n_transactions=len(window_df),
                n_fraud=int(n_fraud),
                fraud_rate=labels.mean(),
                precision=precision_score(labels, preds, zero_division=0),
                recall=recall_score(labels, preds, zero_division=0),
                f1=f1_score(labels, preds, zero_division=0),
                auprc=average_precision_score(labels, scores),
                roc_auc=roc_auc_score(labels, scores),
                alert_rate=preds.mean(),
                mean_score=scores.mean(),
                score_std=scores.std(),
                score_p95=np.percentile(scores, 95),
            )

        self.performance_history.append(snapshot)

        # Run drift detection
        alerts = self.drift_suite.check(
            window_df, scores, labels, preds, timestamp
        )

        # Check for performance degradation
        if self.baseline_auprc and n_fraud > 0:
            auprc_drop = (self.baseline_auprc - snapshot.auprc) / self.baseline_auprc

            if auprc_drop > self.degradation_threshold:
                alert = DriftAlert(
                    timestamp=timestamp,
                    drift_type="degradation",
                    feature_or_metric="auprc",
                    severity="critical" if auprc_drop > 2 * self.degradation_threshold else "warning",
                    value=snapshot.auprc,
                    threshold=self.baseline_auprc * (1 - self.degradation_threshold),
                    message=(f"Model AUPRC degraded {auprc_drop:.1%}: "
                            f"{snapshot.auprc:.4f} vs baseline {self.baseline_auprc:.4f}")
                )
                alerts.append(alert)
                self.degradation_alerts.append(alert)

            # Also check recall separately — missing fraud is costly
            if self.baseline_recall > 0:
                recall_drop = (self.baseline_recall - snapshot.recall) / self.baseline_recall
                if recall_drop > self.degradation_threshold:
                    alert = DriftAlert(
                        timestamp=timestamp,
                        drift_type="degradation",
                        feature_or_metric="recall",
                        severity="critical",
                        value=snapshot.recall,
                        threshold=self.baseline_recall * (1 - self.degradation_threshold),
                        message=(f"Fraud catch rate dropped {recall_drop:.1%}: "
                                f"{snapshot.recall:.4f} vs baseline {self.baseline_recall:.4f}")
                    )
                    alerts.append(alert)
                    self.degradation_alerts.append(alert)

        return snapshot, alerts

    def simulate_production(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate production deployment over the test period.

        Processes data in weekly windows, tracking performance and generating
        alerts. Returns a DataFrame of performance snapshots over time.
        """
        test_df = test_df.sort_values("timestamp").copy()
        start = test_df["timestamp"].min()
        end = test_df["timestamp"].max()

        current = start
        window_delta = pd.Timedelta(days=self.window_size_days)
        window_num = 0

        print(f"\nSimulating production deployment: {start.date()} to {end.date()}")
        print(f"Window size: {self.window_size_days} days")
        print(f"{'='*80}")

        while current < end:
            window_end = current + window_delta
            window = test_df[
                (test_df["timestamp"] >= current) &
                (test_df["timestamp"] < window_end)
            ]

            if len(window) > 0:
                snapshot, alerts = self.process_window(window)
                window_num += 1

                status = "OK"
                if any(a.severity == "critical" for a in alerts):
                    status = "CRITICAL"
                elif any(a.severity == "warning" for a in alerts):
                    status = "WARNING"

                if snapshot and snapshot.n_fraud > 0:
                    print(f"  Week {window_num:2d} [{current.date()}]: "
                          f"AUPRC={snapshot.auprc:.3f} "
                          f"Prec={snapshot.precision:.3f} "
                          f"Rec={snapshot.recall:.3f} "
                          f"FraudRate={snapshot.fraud_rate:.4%} "
                          f"[{status}]")

                for alert in alerts:
                    severity_marker = "!!" if alert.severity == "critical" else "?"
                    print(f"    {severity_marker} {alert.message}")

            current = window_end

        # Build performance history DataFrame
        history_df = pd.DataFrame([
            {
                "window_start": s.window_start,
                "window_end": s.window_end,
                "n_transactions": s.n_transactions,
                "n_fraud": s.n_fraud,
                "fraud_rate": s.fraud_rate,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
                "auprc": s.auprc,
                "roc_auc": s.roc_auc,
                "alert_rate": s.alert_rate,
                "mean_score": s.mean_score,
                "score_std": s.score_std,
                "score_p95": s.score_p95,
            }
            for s in self.performance_history
        ])

        # Summary
        print(f"\n{'='*80}")
        print(f"Production Simulation Summary:")
        print(f"  Total windows: {window_num}")
        print(f"  Drift alerts: {len(self.drift_suite.alert_history)}")
        print(f"  Degradation alerts: {len(self.degradation_alerts)}")
        if self.degradation_alerts:
            first = min(self.degradation_alerts, key=lambda a: a.timestamp)
            print(f"  First degradation detected: {first.timestamp.date()}")
            print(f"    {first.message}")

        return history_df

    def get_retraining_recommendation(self) -> dict:
        """
        Analyze monitoring data and recommend whether to retrain.

        Returns a structured recommendation with:
          - should_retrain: bool
          - urgency: "immediate", "scheduled", "none"
          - reasons: list of specific reasons
          - suggested_strategy: what kind of retraining
        """
        alert_df = self.drift_suite.get_alert_summary()
        n_critical = len([a for a in self.degradation_alerts
                         if a.severity == "critical"])
        n_drift_alerts = len(alert_df) if len(alert_df) > 0 else 0

        # Check trend in AUPRC
        recent_snapshots = self.performance_history[-4:]  # Last 4 windows
        if recent_snapshots and all(s.n_fraud > 0 for s in recent_snapshots):
            recent_auprc = np.mean([s.auprc for s in recent_snapshots])
            auprc_trend = (recent_snapshots[-1].auprc - recent_snapshots[0].auprc)
        else:
            recent_auprc = None
            auprc_trend = 0

        reasons = []
        urgency = "none"

        if n_critical > 2:
            reasons.append(f"{n_critical} critical degradation alerts")
            urgency = "immediate"
        elif n_critical > 0:
            reasons.append(f"{n_critical} critical degradation alert(s)")
            urgency = "scheduled"

        if n_drift_alerts > 5:
            reasons.append(f"{n_drift_alerts} drift alerts indicating data shift")
            if urgency != "immediate":
                urgency = "scheduled"

        if recent_auprc and self.baseline_auprc:
            drop = (self.baseline_auprc - recent_auprc) / self.baseline_auprc
            if drop > 0.2:
                reasons.append(f"AUPRC dropped {drop:.1%} from baseline")
                urgency = "immediate"

        if auprc_trend < -0.05:
            reasons.append("Downward trend in AUPRC over recent windows")
            if urgency == "none":
                urgency = "scheduled"

        should_retrain = urgency != "none"

        strategy = "none"
        if should_retrain:
            if urgency == "immediate":
                strategy = ("Full retrain on recent data with expanded window. "
                           "Consider updating feature engineering to capture "
                           "new fraud patterns (check which features drifted).")
            else:
                strategy = ("Scheduled retrain with incremental data. "
                           "Monitor for another 1-2 windows to confirm trend "
                           "before retraining.")

        return {
            "should_retrain": should_retrain,
            "urgency": urgency,
            "reasons": reasons,
            "suggested_strategy": strategy,
            "recent_auprc": recent_auprc,
            "baseline_auprc": self.baseline_auprc,
        }
