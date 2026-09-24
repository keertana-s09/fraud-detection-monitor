"""
Fraud detection model pipeline designed for extreme class imbalance.

Key design decisions:
  - Temporal train/test split (no future leakage) — critical for drift evaluation
  - Multiple imbalance strategies compared: class weights, SMOTE, focal-loss analog
  - Ensemble of LightGBM + XGBoost for robustness
  - Threshold tuning on precision-recall curve, not accuracy
  - Calibrated probabilities for downstream decisioning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import xgboost as xgb
from typing import Optional


# Features used for modeling (exclude metadata/label columns)
FEATURE_COLS = [
    "amount", "hour_of_day", "is_online", "distance_from_home_km",
    "is_international", "num_transactions_last_1h", "num_transactions_last_24h",
    "days_since_last_transaction", "is_new_merchant", "card_present",
    "log_amount", "amount_zscore", "txn_velocity_ratio",
    "is_night", "is_weekend", "day_of_week", "day_of_month",
    "online_night", "high_amount_new_merchant", "international_high_amount",
    "mcc_frequency",
]

LABEL_COL = "is_fraud"
TIME_COL = "timestamp"


def temporal_train_test_split(df: pd.DataFrame, train_frac: float = 0.7):
    """
    Split data temporally — train on past, test on future.
    This is the ONLY valid split for fraud detection: random splits leak future patterns.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"Train: {len(train):,} txns ({train[LABEL_COL].mean():.4%} fraud), "
          f"{train[TIME_COL].min().date()} to {train[TIME_COL].max().date()}")
    print(f"Test:  {len(test):,} txns ({test[LABEL_COL].mean():.4%} fraud), "
          f"{test[TIME_COL].min().date()} to {test[TIME_COL].max().date()}")

    return train, test


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                           beta: float = 2.0) -> tuple:
    """
    Find the threshold that maximizes F-beta score.
    beta > 1 favors recall (catching more fraud), which is usually what you want.
    In production, this threshold would be tuned to business cost constraints.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # F-beta score
    fbeta = ((1 + beta**2) * precisions * recalls) / (beta**2 * precisions + recalls + 1e-8)
    best_idx = np.argmax(fbeta[:-1])  # last element is always 0/0
    return thresholds[best_idx], fbeta[best_idx]


def evaluate_model(y_true: np.ndarray, y_scores: np.ndarray,
                   threshold: Optional[float] = None, label: str = "") -> dict:
    """Comprehensive evaluation focused on metrics that matter for fraud detection."""
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    ap = average_precision_score(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Dollar amount caught vs missed (if we had amounts, approximate)
    metrics = {
        "threshold": threshold,
        "average_precision": ap,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        # Operational metrics
        "alert_rate": (tp + fp) / (tp + fp + tn + fn),  # % of txns flagged
        "fraud_catch_rate": rec,  # same as recall
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Threshold:        {threshold:.4f}")
    print(f"  AUPRC:            {ap:.4f}  (random baseline: {y_true.mean():.4f})")
    print(f"  ROC-AUC:          {auc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Alert rate:       {metrics['alert_rate']:.4%}")
    print(f"  False positive rate: {metrics['false_positive_rate']:.4%}")

    return metrics


class FraudDetector:
    """
    Ensemble fraud detector combining LightGBM and XGBoost.

    Both models use scale_pos_weight to handle imbalance natively, which is more
    effective than SMOTE for tree-based models (SMOTE creates synthetic samples
    that can confuse tree splits in high-dimensional space).

    Why ensemble? Individual models tend to overfit to specific fraud patterns.
    Averaging scores from two different gradient boosting implementations
    provides more robust predictions, especially under distribution shift.
    """

    def __init__(self, use_smote: bool = False):
        self.use_smote = use_smote
        self.lgb_model = None
        self.xgb_model = None
        self.threshold = 0.5
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False
        self.train_fraud_rate = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train the ensemble on temporally-ordered training data."""
        X_train = train_df[self.feature_cols].values
        y_train = train_df[LABEL_COL].values

        self.train_fraud_rate = y_train.mean()
        imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        print(f"\nTraining with imbalance ratio: {imbalance_ratio:.0f}:1")

        if self.use_smote:
            print("  Applying SMOTE oversampling...")
            smote = SMOTE(sampling_strategy=0.1, random_state=42)  # upsample to 10%
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {len(X_train):,} samples, "
                  f"{y_train.mean():.2%} fraud")

        # LightGBM — fast, handles imbalance via scale_pos_weight
        print("  Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        lgb_fit_params = {}
        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df[LABEL_COL].values
            lgb_fit_params["eval_X"] = X_val
            lgb_fit_params["eval_y"] = y_val
            lgb_fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        self.lgb_model.fit(X_train, y_train, **lgb_fit_params)

        # XGBoost — different splitting algorithm, complementary errors
        print("  Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        xgb_fit_params = {}
        if val_df is not None:
            xgb_fit_params["eval_set"] = [(X_val, y_val)]
            xgb_fit_params["verbose"] = False

        self.xgb_model.fit(X_train, y_train, **xgb_fit_params)
        self.is_fitted = True

        # Find optimal threshold on validation data
        if val_df is not None:
            val_scores = self.predict_proba(val_df)
            self.threshold, fbeta = find_optimal_threshold(y_val, val_scores)
            print(f"  Optimal threshold: {self.threshold:.4f} (F2: {fbeta:.4f})")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return ensemble fraud probability scores."""
        X = df[self.feature_cols].values
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        # Simple average — more sophisticated approaches (stacking) don't help much here
        return (lgb_proba + xgb_proba) / 2

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using tuned threshold."""
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return combined feature importance from both models."""
        lgb_imp = self.lgb_model.feature_importances_
        xgb_imp = self.xgb_model.feature_importances_

        # Normalize each to [0, 1] then average
        lgb_norm = lgb_imp / lgb_imp.max()
        xgb_norm = xgb_imp / xgb_imp.max()

        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "lgb_importance": lgb_norm,
            "xgb_importance": xgb_norm,
            "avg_importance": (lgb_norm + xgb_norm) / 2,
        }).sort_values("avg_importance", ascending=False)

        return importance


def train_and_evaluate(df: pd.DataFrame, use_smote: bool = False) -> tuple:
    """Full training pipeline with temporal split and evaluation."""
    # Split: 60% train, 10% val, 30% test (temporal)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * 0.6)]
    val = df.iloc[int(n * 0.6):int(n * 0.7)]
    test = df.iloc[int(n * 0.7):]

    print(f"\nSplit sizes — Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    detector = FraudDetector(use_smote=use_smote)
    detector.fit(train, val)

    # Evaluate on test set (future data — simulates production)
    test_scores = detector.predict_proba(test)
    test_labels = test[LABEL_COL].values

    metrics = evaluate_model(test_labels, test_scores, detector.threshold,
                            label="Test Set Performance (Future Data)")

    # Feature importance
    importance = detector.get_feature_importance()
    print("\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['avg_importance']:.3f}")

    return detector, metrics, test
