"""
FastAPI backend for the fraud detection monitoring dashboard.

Users can either upload their own transaction CSV or use generated demo data.
The pipeline trains a model, runs production monitoring, and serves results.

Endpoints:
  GET  /api/status             — whether pipeline has run yet
  POST /api/upload             — upload a transaction CSV, run full pipeline
  POST /api/demo               — run pipeline on simulated demo data
  GET  /api/overview           — high-level stats
  GET  /api/metrics            — per-window performance metrics over time
  GET  /api/alerts             — drift and degradation alerts
  GET  /api/drift-scores       — PSI scores per window
  GET  /api/feature-importance — model feature importance
  GET  /api/transactions       — sample transactions with scores (paginated)
  GET  /api/fraud-type-mix     — fraud type composition over time
  POST /api/retrain            — retrain on recent data, return before/after
  GET  /api/column-map         — expected vs detected columns for upload validation
"""

import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score, recall_score

from data_simulator import TransactionSimulator, DriftConfig, add_engineered_features
from model import FraudDetector, FEATURE_COLS, LABEL_COL, TIME_COL
from drift_detector import PSIDetector
from monitor import ProductionMonitor

# ── Global state ─────────────────────────────────────────────────────────────
state = {}

app = FastAPI(title="Fraud Detection Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Column mapping for user uploads ──────────────────────────────────────────
# Users' CSVs won't have our exact column names. This maps common alternatives.
COLUMN_ALIASES = {
    # Our internal name → list of names we'll accept from user CSVs
    "amount": ["amount", "amt", "transaction_amount", "trans_amt", "value", "txn_amount"],
    "timestamp": ["timestamp", "date", "datetime", "time", "trans_date", "transaction_date",
                   "trans_date_trans_time"],
    "is_fraud": ["is_fraud", "fraud", "label", "class", "target", "isFraud"],
    "merchant_category": ["merchant_category", "category", "merchant", "mcc",
                          "merchant_name", "merchant_type"],
    "hour_of_day": ["hour_of_day", "hour", "trans_hour"],
    "is_online": ["is_online", "online", "card_not_present", "cnp"],
    "distance_from_home_km": ["distance_from_home_km", "distance", "distance_from_home",
                               "dist_from_home"],
    "is_international": ["is_international", "international", "foreign"],
    "num_transactions_last_1h": ["num_transactions_last_1h", "trans_1h", "velocity_1h",
                                  "txn_count_1h"],
    "num_transactions_last_24h": ["num_transactions_last_24h", "trans_24h", "velocity_24h",
                                   "txn_count_24h"],
    "days_since_last_transaction": ["days_since_last_transaction", "days_since_last",
                                     "recency", "days_since_prev"],
    "is_new_merchant": ["is_new_merchant", "new_merchant", "first_time_merchant"],
    "card_present": ["card_present", "card_present_flag", "in_store"],
    "customer_id": ["customer_id", "cust_id", "user_id", "account_id", "cc_num",
                     "card_number", "card_id"],
    "fraud_type": ["fraud_type", "fraud_category", "fraud_class"],
}


def resolve_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Map user CSV columns to our internal names.
    Returns (remapped_df, mapping_report).
    """
    col_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    missing = []

    for internal_name, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias.lower() in col_lower:
                original = col_lower[alias.lower()]
                if original != internal_name:
                    df = df.rename(columns={original: internal_name})
                mapping[internal_name] = original
                found = True
                break
        if not found:
            missing.append(internal_name)

    return df, {"mapped": mapping, "missing": missing}


def synthesize_missing_features(df: pd.DataFrame, missing: list) -> pd.DataFrame:
    """
    For columns the user didn't provide, generate reasonable defaults
    so the model can still run. Better than refusing the upload.
    """
    rng = np.random.default_rng(42)
    n = len(df)

    if "timestamp" in missing and "timestamp" not in df.columns:
        # If no timestamp, create a synthetic one spanning 365 days
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="min")

    if "customer_id" in missing and "customer_id" not in df.columns:
        df["customer_id"] = rng.integers(0, max(n // 10, 100), size=n)

    if "hour_of_day" in missing and "hour_of_day" not in df.columns:
        if "timestamp" in df.columns:
            df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
        else:
            df["hour_of_day"] = rng.integers(0, 24, size=n)

    if "merchant_category" in missing and "merchant_category" not in df.columns:
        cats = ["online_retail", "grocery", "gas_station", "restaurant",
                "electronics", "travel", "subscription"]
        df["merchant_category"] = rng.choice(cats, size=n)

    if "is_online" in missing and "is_online" not in df.columns:
        df["is_online"] = rng.integers(0, 2, size=n)

    if "distance_from_home_km" in missing and "distance_from_home_km" not in df.columns:
        df["distance_from_home_km"] = np.abs(rng.exponential(5, size=n)).round(2)

    if "is_international" in missing and "is_international" not in df.columns:
        df["is_international"] = (rng.random(n) < 0.03).astype(int)

    if "num_transactions_last_1h" in missing and "num_transactions_last_1h" not in df.columns:
        df["num_transactions_last_1h"] = rng.poisson(0.5, size=n)

    if "num_transactions_last_24h" in missing and "num_transactions_last_24h" not in df.columns:
        df["num_transactions_last_24h"] = rng.poisson(3, size=n)

    if "days_since_last_transaction" in missing and "days_since_last_transaction" not in df.columns:
        df["days_since_last_transaction"] = np.abs(rng.exponential(1.5, size=n)).round(1)

    if "is_new_merchant" in missing and "is_new_merchant" not in df.columns:
        df["is_new_merchant"] = (rng.random(n) < 0.15).astype(int)

    if "card_present" in missing and "card_present" not in df.columns:
        if "is_online" in df.columns:
            df["card_present"] = 1 - df["is_online"]
        else:
            df["card_present"] = rng.integers(0, 2, size=n)

    if "fraud_type" in missing and "fraud_type" not in df.columns:
        df["fraud_type"] = "none"
        if "is_fraud" in df.columns:
            df.loc[df["is_fraud"] == 1, "fraud_type"] = "unknown"

    return df


def run_pipeline(df: pd.DataFrame, data_source: str = "upload"):
    """
    Core pipeline: takes a DataFrame, trains model, runs monitoring, populates state.
    Works identically for uploaded data and demo data.
    """
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    # Ensure is_fraud is int
    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)

    # Add engineered features
    df = add_engineered_features(df)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    if n < 500:
        raise ValueError(f"Dataset too small ({n} rows). Need at least 500 transactions.")

    fraud_count = df["is_fraud"].sum()
    if fraud_count == 0:
        raise ValueError("No fraud labels found. The dataset needs an 'is_fraud' column with at least some 1s.")

    train_df = df.iloc[:int(n * 0.6)]
    val_df = df.iloc[int(n * 0.6):int(n * 0.7)]
    prod_df = df.iloc[int(n * 0.7):]

    # Check that train and val have some fraud
    if train_df["is_fraud"].sum() < 5:
        raise ValueError("Not enough fraud in the training period (first 60% of data). Need at least 5 fraud cases.")
    if val_df["is_fraud"].sum() < 2:
        raise ValueError("Not enough fraud in the validation period (60-70% of data).")

    # Train model
    detector = FraudDetector(use_smote=False)
    detector.fit(train_df, val_df)

    # Run production monitoring
    # Adaptive window size: at least 3 windows in the prod period
    prod_days = (prod_df[TIME_COL].max() - prod_df[TIME_COL].min()).days
    window_days = max(1, min(7, prod_days // 4))

    monitor = ProductionMonitor(
        model=detector,
        feature_cols=FEATURE_COLS,
        window_size_days=window_days,
        degradation_threshold=0.15,
    )
    monitor.set_baseline(train_df, val_df)
    history_df = monitor.simulate_production(prod_df)

    # PSI timeline
    psi_detector = PSIDetector()
    train_scores = detector.predict_proba(train_df)
    psi_detector.set_reference(train_scores)

    psi_timeline = []
    prod_sorted = prod_df.sort_values(TIME_COL)
    window_delta = pd.Timedelta(days=window_days)
    current = prod_sorted[TIME_COL].min()
    end = prod_sorted[TIME_COL].max()
    while current < end:
        window = prod_sorted[
            (prod_sorted[TIME_COL] >= current) &
            (prod_sorted[TIME_COL] < current + window_delta)
        ]
        if len(window) > 0:
            scores = detector.predict_proba(window)
            psi_val = psi_detector.calculate(scores)
            psi_timeline.append({
                "date": current.isoformat(),
                "psi": round(float(psi_val), 4),
                "warning_threshold": 0.1,
                "critical_threshold": 0.25,
            })
        current += window_delta

    # Sample transactions for the table
    sample_size = min(500, len(prod_df))
    sample_df = prod_df.sample(n=sample_size, random_state=42).sort_values(TIME_COL)
    sample_scores = detector.predict_proba(sample_df)
    sample_preds = detector.predict(sample_df)

    transactions = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        transactions.append({
            "id": int(row.get("transaction_id", i)),
            "timestamp": row[TIME_COL].isoformat(),
            "amount": round(float(row["amount"]), 2),
            "merchant_category": str(row.get("merchant_category", "unknown")),
            "is_online": int(row.get("is_online", 0)),
            "is_international": int(row.get("is_international", 0)),
            "is_new_merchant": int(row.get("is_new_merchant", 0)),
            "fraud_score": round(float(sample_scores[i]), 4),
            "predicted_fraud": int(sample_preds[i]),
            "actual_fraud": int(row["is_fraud"]),
            "fraud_type": str(row.get("fraud_type", "unknown")),
        })

    # Fraud type mix over time
    fraud_only = df[df["is_fraud"] == 1].copy()
    fraud_only["month"] = fraud_only[TIME_COL].dt.to_period("M").astype(str)
    type_mix = []
    fraud_types_found = fraud_only["fraud_type"].unique().tolist()
    fraud_types_found = [ft for ft in fraud_types_found if ft not in ("none", "unknown", "")]
    if fraud_types_found:
        for month, group in fraud_only.groupby("month"):
            counts = group["fraud_type"].value_counts()
            total = counts.sum()
            entry = {"month": month}
            for ftype in fraud_types_found:
                entry[ftype] = round(float(counts.get(ftype, 0) / total * 100), 1)
            type_mix.append(entry)

    # Feature importance
    importance = detector.get_feature_importance()
    feature_imp = [
        {"feature": row["feature"], "importance": round(float(row["avg_importance"]), 4)}
        for _, row in importance.head(15).iterrows()
    ]

    # Alerts
    alerts_df = monitor.drift_suite.get_alert_summary()
    alerts = []
    if len(alerts_df) > 0:
        for _, row in alerts_df.iterrows():
            alerts.append({
                "timestamp": row["timestamp"].isoformat(),
                "type": row["type"],
                "feature": row["feature"],
                "severity": row["severity"],
                "value": round(float(row["value"]), 4),
                "message": row["message"],
            })
    for a in monitor.degradation_alerts:
        alerts.append({
            "timestamp": a.timestamp.isoformat(),
            "type": a.drift_type,
            "feature": a.feature_or_metric,
            "severity": a.severity,
            "value": round(float(a.value), 4),
            "message": a.message,
        })

    # Metrics timeline
    metrics_timeline = []
    for _, row in history_df.iterrows():
        if row["n_fraud"] > 0:
            metrics_timeline.append({
                "date": row["window_start"].isoformat(),
                "auprc": round(float(row["auprc"]), 4),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1": round(float(row["f1"]), 4),
                "fraud_rate": round(float(row["fraud_rate"]) * 100, 4),
                "alert_rate": round(float(row["alert_rate"]) * 100, 4),
                "n_transactions": int(row["n_transactions"]),
                "n_fraud": int(row["n_fraud"]),
                "mean_score": round(float(row["mean_score"]), 4),
            })

    rec = monitor.get_retraining_recommendation()

    # Populate state
    state["ready"] = True
    state["data_source"] = data_source
    state["overview"] = {
        "total_transactions": len(df),
        "total_fraud": int(df["is_fraud"].sum()),
        "fraud_rate": round(float(df["is_fraud"].mean()) * 100, 3),
        "model_type": "LightGBM + XGBoost Ensemble",
        "baseline_auprc": round(float(monitor.baseline_auprc), 4),
        "baseline_precision": round(float(monitor.baseline_precision), 4),
        "baseline_recall": round(float(monitor.baseline_recall), 4),
        "total_alerts": len(alerts),
        "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
        "retrain_urgency": rec["urgency"],
        "retrain_reasons": rec["reasons"],
        "train_period": f"{train_df[TIME_COL].min().date()} to {train_df[TIME_COL].max().date()}",
        "prod_period": f"{prod_df[TIME_COL].min().date()} to {prod_df[TIME_COL].max().date()}",
        "data_source": data_source,
    }
    state["metrics"] = metrics_timeline
    state["alerts"] = alerts
    state["psi_timeline"] = psi_timeline
    state["feature_importance"] = feature_imp
    state["transactions"] = transactions
    state["fraud_type_mix"] = type_mix
    state["fraud_types_found"] = fraud_types_found

    state["_df"] = df
    state["_train_df"] = train_df
    state["_val_df"] = val_df
    state["_prod_df"] = prod_df
    state["_detector"] = detector
    state["_monitor"] = monitor

    print(f"Pipeline complete ({data_source}). {len(df):,} transactions, "
          f"{fraud_count} fraud, {len(alerts)} alerts.")


# ── API Routes ───────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Check if the pipeline has been run yet."""
    return {
        "ready": state.get("ready", False),
        "data_source": state.get("data_source"),
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a transaction CSV. Required: amount, is_fraud.
    Recommended: timestamp, customer_id, merchant_category.
    Everything else is synthesized if missing.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a CSV")

    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(400, "File too large (max 100MB)")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if len(df) < 500:
        raise HTTPException(400, f"Dataset too small ({len(df)} rows). Need at least 500 transactions.")

    # Map columns
    df, col_report = resolve_columns(df)

    # Check for required columns
    if "amount" not in df.columns and "amount" in col_report["missing"]:
        raise HTTPException(400, "CSV must have an 'amount' column (transaction amount).")
    if "is_fraud" not in df.columns and "is_fraud" in col_report["missing"]:
        raise HTTPException(400, "CSV must have an 'is_fraud' column (0/1 fraud label).")

    # Fill in missing features
    df = synthesize_missing_features(df, col_report["missing"])

    try:
        run_pipeline(df, data_source=f"upload ({file.filename})")
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "status": "ok",
        "rows": len(df),
        "fraud_count": int(df["is_fraud"].sum()),
        "columns_mapped": col_report["mapped"],
        "columns_synthesized": [c for c in col_report["missing"]
                                if c not in ("amount", "is_fraud")],
    }


@app.post("/api/demo")
def load_demo():
    """Run the pipeline on simulated demo data with concept drift."""
    sim = TransactionSimulator(seed=42, drift_config=DriftConfig())
    df = sim.generate(n_transactions=150_000, n_customers=10_000, n_days=365)
    run_pipeline(df, data_source="demo (simulated)")
    return {"status": "ok", "rows": len(df), "fraud_count": int(df["is_fraud"].sum())}


@app.get("/api/overview")
def get_overview():
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet. Upload a CSV or load demo data.")
    return state["overview"]


@app.get("/api/metrics")
def get_metrics():
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    return state["metrics"]


@app.get("/api/alerts")
def get_alerts(severity: str = None):
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    alerts = state["alerts"]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    return alerts


@app.get("/api/drift-scores")
def get_drift_scores():
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    return state["psi_timeline"]


@app.get("/api/feature-importance")
def get_feature_importance():
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    return state["feature_importance"]


@app.get("/api/transactions")
def get_transactions(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=200),
    fraud_only: bool = False,
):
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    txns = state["transactions"]
    if fraud_only:
        txns = [t for t in txns if t["actual_fraud"] == 1]
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "transactions": txns[start:end],
        "total": len(txns),
        "page": page,
        "per_page": per_page,
    }


@app.get("/api/fraud-type-mix")
def get_fraud_type_mix():
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")
    return state["fraud_type_mix"]


@app.post("/api/retrain")
def retrain_model():
    """Retrain on recent data, return before/after comparison."""
    if not state.get("ready"):
        raise HTTPException(425, "Pipeline hasn't run yet.")

    df = state["_df"]
    prod_df = state["_prod_df"]
    old_detector = state["_detector"]

    late_prod = prod_df.tail(int(len(prod_df) * 0.5))
    old_scores = old_detector.predict_proba(late_prod)
    old_labels = late_prod[LABEL_COL].values

    old_auprc = average_precision_score(old_labels, old_scores)
    old_preds = old_detector.predict(late_prod)
    old_precision = precision_score(old_labels, old_preds, zero_division=0)
    old_recall = recall_score(old_labels, old_preds, zero_division=0)

    n = len(df)
    new_train = df.iloc[:int(n * 0.85)]
    new_val = df.iloc[int(n * 0.85):int(n * 0.9)]

    new_detector = FraudDetector(use_smote=False)
    new_detector.fit(new_train, new_val)

    new_scores = new_detector.predict_proba(late_prod)
    new_auprc = average_precision_score(old_labels, new_scores)
    new_preds = new_detector.predict(late_prod)
    new_precision = precision_score(old_labels, new_preds, zero_division=0)
    new_recall = recall_score(old_labels, new_preds, zero_division=0)

    state["_detector"] = new_detector
    state["overview"]["retrain_urgency"] = "none"
    state["overview"]["retrain_reasons"] = ["Model was just retrained"]

    importance = new_detector.get_feature_importance()
    state["feature_importance"] = [
        {"feature": row["feature"], "importance": round(float(row["avg_importance"]), 4)}
        for _, row in importance.head(15).iterrows()
    ]

    return {
        "status": "retrained",
        "evaluation_period": f"{late_prod[TIME_COL].min().date()} to {late_prod[TIME_COL].max().date()}",
        "before": {
            "auprc": round(float(old_auprc), 4),
            "precision": round(float(old_precision), 4),
            "recall": round(float(old_recall), 4),
            "train_period": state["overview"]["train_period"],
        },
        "after": {
            "auprc": round(float(new_auprc), 4),
            "precision": round(float(new_precision), 4),
            "recall": round(float(new_recall), 4),
            "train_period": f"{new_train[TIME_COL].min().date()} to {new_train[TIME_COL].max().date()}",
        },
        "improvement": {
            "auprc_delta": round(float(new_auprc - old_auprc), 4),
            "precision_delta": round(float(new_precision - old_precision), 4),
            "recall_delta": round(float(new_recall - old_recall), 4),
        },
    }


# ── Serve React frontend ─────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
