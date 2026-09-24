"""
Concept drift detection for fraud models.

In production fraud detection, the model degrades silently because:
  1. Fraudsters adapt (adversarial drift) — they learn what gets caught
  2. Customer behavior changes (natural drift) — seasonal patterns, new products
  3. Data pipeline changes (sudden drift) — upstream schema changes, new merchants

This module implements three complementary drift detection approaches:
  - Population Stability Index (PSI): detects shifts in score distributions
  - Kolmogorov-Smirnov test: detects shifts in individual feature distributions
  - ADWIN (Adaptive Windowing): detects changes in streaming error rates
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class DriftAlert:
    """A detected drift event."""
    timestamp: pd.Timestamp
    drift_type: str  # "psi", "ks", "adwin", "label_drift"
    feature_or_metric: str
    severity: str  # "warning", "critical"
    value: float
    threshold: float
    message: str


class PSIDetector:
    """
    Population Stability Index — compares two score/feature distributions.

    PSI < 0.1:  No significant shift
    PSI 0.1-0.25: Moderate shift — investigate
    PSI > 0.25: Significant shift — retrain

    This is the industry standard for monitoring model score distributions.
    It's symmetric-ish, interpretable, and works well with binned continuous data.
    """

    def __init__(self, n_bins: int = 10, warning_threshold: float = 0.1,
                 critical_threshold: float = 0.25):
        self.n_bins = n_bins
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.reference_bins = None
        self.bin_edges = None

    def set_reference(self, values: np.ndarray):
        """Set reference distribution from training/baseline data."""
        # Use quantile-based binning for robustness
        self.bin_edges = np.percentile(
            values, np.linspace(0, 100, self.n_bins + 1)
        )
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        # Ensure unique edges
        self.bin_edges = np.unique(self.bin_edges)
        counts = np.histogram(values, bins=self.bin_edges)[0]
        self.reference_bins = counts / counts.sum()
        # Clip to avoid division by zero
        self.reference_bins = np.clip(self.reference_bins, 1e-6, None)

    def calculate(self, values: np.ndarray) -> float:
        """Calculate PSI between reference and new distribution."""
        if self.reference_bins is None:
            raise ValueError("Must call set_reference first")

        counts = np.histogram(values, bins=self.bin_edges)[0]
        new_bins = counts / max(counts.sum(), 1)
        new_bins = np.clip(new_bins, 1e-6, None)

        # PSI = sum((new - ref) * ln(new / ref))
        psi = np.sum((new_bins - self.reference_bins) *
                      np.log(new_bins / self.reference_bins))
        return psi

    def check(self, values: np.ndarray, feature_name: str = "score",
              timestamp: pd.Timestamp = None) -> Optional[DriftAlert]:
        """Check for distribution shift and return alert if detected."""
        psi = self.calculate(values)

        if psi >= self.critical_threshold:
            return DriftAlert(
                timestamp=timestamp or pd.Timestamp.now(),
                drift_type="psi",
                feature_or_metric=feature_name,
                severity="critical",
                value=psi,
                threshold=self.critical_threshold,
                message=f"Critical distribution shift in {feature_name}: PSI={psi:.4f}"
            )
        elif psi >= self.warning_threshold:
            return DriftAlert(
                timestamp=timestamp or pd.Timestamp.now(),
                drift_type="psi",
                feature_or_metric=feature_name,
                severity="warning",
                value=psi,
                threshold=self.warning_threshold,
                message=f"Moderate distribution shift in {feature_name}: PSI={psi:.4f}"
            )
        return None


class KSFeatureDriftDetector:
    """
    Kolmogorov-Smirnov test for individual feature drift.

    Compares each feature's distribution in a new window against the training
    distribution. More sensitive than PSI for detecting subtle shifts in
    specific features (e.g., transaction amounts creeping up).
    """

    def __init__(self, p_value_threshold: float = 0.001):
        """
        p_value_threshold is deliberately very low (0.001 not 0.05) because:
        - With large samples, KS detects tiny irrelevant differences at p=0.05
        - We only want to alert on meaningful shifts, not statistical noise
        """
        self.p_value_threshold = p_value_threshold
        self.reference_distributions = {}
        self._rng = np.random.default_rng(42)

    def set_reference(self, df: pd.DataFrame, feature_cols: list,
                      max_samples: int = 5000):
        """Store reference distributions for each feature (sampled for speed)."""
        rng = np.random.default_rng(42)
        for col in feature_cols:
            vals = df[col].dropna().values
            if len(vals) > max_samples:
                vals = rng.choice(vals, max_samples, replace=False)
            self.reference_distributions[col] = vals

    def check_features(self, df: pd.DataFrame,
                       timestamp: pd.Timestamp = None) -> list:
        """Test each feature for distribution shift."""
        alerts = []
        for col, ref_values in self.reference_distributions.items():
            if col not in df.columns:
                continue
            new_values = df[col].dropna().values
            if len(new_values) > 5000:
                new_values = new_values[self._rng.choice(len(new_values), 5000, replace=False)]
            if len(new_values) < 30:
                continue

            stat, p_value = stats.ks_2samp(ref_values, new_values)

            if p_value < self.p_value_threshold:
                severity = "critical" if p_value < self.p_value_threshold / 10 else "warning"
                alerts.append(DriftAlert(
                    timestamp=timestamp or pd.Timestamp.now(),
                    drift_type="ks",
                    feature_or_metric=col,
                    severity=severity,
                    value=stat,
                    threshold=self.p_value_threshold,
                    message=f"Feature drift in '{col}': KS={stat:.4f}, p={p_value:.2e}"
                ))

        return alerts


class ADWINDetector:
    """
    Simplified Adaptive Windowing for streaming error rate monitoring.

    Compares the error rate of the current batch against the previous batch
    using a Hoeffding bound. Lightweight alternative to full ADWIN that
    works on batch-level aggregates rather than individual elements.

    Based on the idea from Bifet & Gavaldà (2007), simplified for batch mode.
    """

    def __init__(self, delta: float = 0.01):
        self.delta = delta
        self.prev_errors = None
        self.drift_detected = False

    def check_batch(self, errors: np.ndarray) -> bool:
        """
        Check a batch of errors against the previous batch.
        Returns True if drift is detected.
        """
        self.drift_detected = False

        if self.prev_errors is None or len(errors) < 30:
            self.prev_errors = errors
            return False

        n1, n2 = len(self.prev_errors), len(errors)
        mean1, mean2 = self.prev_errors.mean(), errors.mean()

        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))

        if abs(mean1 - mean2) >= epsilon:
            self.drift_detected = True
            self.prev_errors = errors
            return True

        self.prev_errors = errors
        return False

    def reset(self):
        self.prev_errors = None
        self.drift_detected = False


class LabelDriftDetector:
    """
    Monitors the actual fraud rate over time.

    In practice, fraud labels arrive with a delay (chargebacks take 30-90 days).
    This detector works on confirmed labels as they arrive, checking if the
    underlying fraud rate has shifted — which would mean the model's priors
    are stale even if features haven't changed.
    """

    def __init__(self, window_size: int = 5000, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.reference_rate = None

    def set_reference(self, fraud_rate: float):
        """Set the expected fraud rate from training data."""
        self.reference_rate = fraud_rate

    def check(self, labels: np.ndarray,
              timestamp: pd.Timestamp = None) -> Optional[DriftAlert]:
        """Check if observed fraud rate has shifted from reference."""
        if self.reference_rate is None or len(labels) < 100:
            return None

        observed_rate = labels.mean()
        # Z-test for proportion
        se = np.sqrt(self.reference_rate * (1 - self.reference_rate) / len(labels))
        if se < 1e-8:
            return None
        z = (observed_rate - self.reference_rate) / se

        if abs(z) > self.z_threshold:
            direction = "increased" if z > 0 else "decreased"
            severity = "critical" if abs(z) > self.z_threshold * 1.5 else "warning"
            return DriftAlert(
                timestamp=timestamp or pd.Timestamp.now(),
                drift_type="label_drift",
                feature_or_metric="fraud_rate",
                severity=severity,
                value=observed_rate,
                threshold=self.reference_rate,
                message=(f"Fraud rate {direction}: {observed_rate:.4%} vs "
                        f"reference {self.reference_rate:.4%} (z={z:.2f})")
            )
        return None


class DriftMonitoringSuite:
    """
    Unified drift monitoring that combines all detectors.

    Usage:
        suite = DriftMonitoringSuite(feature_cols)
        suite.set_reference(train_df, train_scores)

        # In production loop:
        for window in data_stream:
            scores = model.predict_proba(window)
            alerts = suite.check(window, scores, labels_if_available)
            if alerts:
                trigger_investigation(alerts)
    """

    def __init__(self, feature_cols: list, ks_features: int = 8):
        self.feature_cols = feature_cols
        self.ks_features = ks_features  # Only run KS on top N features
        self.score_psi = PSIDetector()
        self.ks_detector = KSFeatureDriftDetector()
        self.adwin = ADWINDetector(delta=0.01)
        self.label_drift = LabelDriftDetector()
        self.alert_history = []

    def set_reference(self, train_df: pd.DataFrame, train_scores: np.ndarray):
        """Initialize all detectors with training data distributions."""
        self.score_psi.set_reference(train_scores)
        # Only monitor a subset of features for KS (the most variable ones)
        ks_cols = self.feature_cols[:self.ks_features]
        self.ks_detector.set_reference(train_df, ks_cols)
        self.label_drift.set_reference(train_df["is_fraud"].mean())

    def check(self, window_df: pd.DataFrame, scores: np.ndarray,
              labels: Optional[np.ndarray] = None,
              predictions: Optional[np.ndarray] = None,
              timestamp: pd.Timestamp = None) -> list:
        """
        Run all drift checks on a data window.

        Args:
            window_df: Recent transaction data
            scores: Model probability scores for this window
            labels: Ground truth labels (may be None if not yet available)
            predictions: Binary predictions (for error rate monitoring)
            timestamp: Timestamp for this check

        Returns:
            List of DriftAlert objects
        """
        alerts = []

        # 1. Score distribution shift (always available — no labels needed)
        score_alert = self.score_psi.check(scores, "model_score", timestamp)
        if score_alert:
            alerts.append(score_alert)

        # 2. Feature distribution shifts
        feature_alerts = self.ks_detector.check_features(window_df, timestamp)
        alerts.extend(feature_alerts)

        # 3. Error rate change detection (needs predictions and labels)
        if labels is not None and predictions is not None:
            errors = (predictions != labels).astype(float)
            if self.adwin.check_batch(errors):
                alerts.append(DriftAlert(
                    timestamp=timestamp or pd.Timestamp.now(),
                    drift_type="adwin",
                    feature_or_metric="error_rate",
                    severity="critical",
                    value=errors.mean(),
                    threshold=0.0,
                    message=f"ADWIN detected change in model error rate: {errors.mean():.4f}"
                ))

        # 4. Label distribution shift (needs labels)
        if labels is not None:
            label_alert = self.label_drift.check(labels, timestamp)
            if label_alert:
                alerts.append(label_alert)

        self.alert_history.extend(alerts)
        return alerts

    def get_alert_summary(self) -> pd.DataFrame:
        """Return all historical alerts as a DataFrame."""
        if not self.alert_history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "type": a.drift_type,
                "feature": a.feature_or_metric,
                "severity": a.severity,
                "value": a.value,
                "message": a.message,
            }
            for a in self.alert_history
        ])
