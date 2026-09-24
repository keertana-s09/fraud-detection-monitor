# Fraud Detection Monitor

Real-time fraud detection system with concept drift monitoring. Trains an ensemble ML model on transaction data, simulates production deployment, and surfaces silent model degradation through an interactive dashboard.

**Why this exists:** Most fraud models degrade silently in production because fraudsters adapt and customer behavior shifts. This project demonstrates detecting that degradation before fraud losses spike — the part that matters in production and that most portfolio projects skip.

## Dashboard

Upload your own transaction CSV or use simulated demo data with built-in concept drift.

**What the dashboard shows:**
- Model performance (AUPRC, precision, recall) tracked over weekly windows
- PSI drift score timeline with warning/critical thresholds
- Filterable alert panel for drift and degradation events
- Fraud type composition showing how attack patterns evolve
- Transaction stream with fraud scores, predictions vs. actuals
- One-click retrain with before/after metrics comparison

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  Upload Page → Performance Charts → Alert Panel      │
│  Drift Timeline → Transaction Table → Retrain Loop   │
└──────────────────────┬──────────────────────────────┘
                       │ /api
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                      │
│  POST /upload  POST /demo  POST /retrain             │
│  GET /metrics  GET /alerts  GET /drift-scores        │
└──────┬───────────────┬───────────────┬──────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   Model     │ │   Drift     │ │  Production │
│  Pipeline   │ │  Detectors  │ │   Monitor   │
│             │ │             │ │             │
│ LightGBM +  │ │ PSI (score) │ │ Weekly perf │
│ XGBoost     │ │ KS (feature)│ │ Degradation │
│ ensemble    │ │ ADWIN (err) │ │ alerts      │
│             │ │ Label drift │ │ Retrain rec │
└─────────────┘ └─────────────┘ └─────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..

# Run
./start.sh
# Open http://localhost:8000
```

## How It Works

### Class Imbalance (<1% fraud)

The model uses `scale_pos_weight` in both LightGBM and XGBoost to tell the training algorithm that missing a fraud case costs ~178x more than a false alarm. The threshold is tuned on the F2 score (favoring recall over precision), and evaluation uses AUPRC instead of accuracy — because a model that predicts "not fraud" for everything gets 99%+ accuracy but catches nothing.

### Concept Drift

The simulator generates four fraud types that evolve over a year:

| Fraud Type | Early | Late | Behavior |
|---|---|---|---|
| CNP (card-not-present) | 60% of fraud | 15% | Declines as 3D Secure adoption grows |
| Account Takeover | 25% | 45% | Grows as phishing scales |
| Bust-out | 15% | 10% | Periodic bursts from organized rings |
| Micro-charge testing | 0% | 30% | New attack vector emerges mid-year |

The model is trained on early data. As fraud patterns shift, the drift detectors catch it.

### Drift Detection

Four complementary detectors, because no single one catches everything:

- **PSI** — Compares model score distributions against training baseline. Works without labels (critical since chargebacks arrive 30-90 days late).
- **KS tests** — Tests individual feature distributions for shift. Tells you *which* inputs changed.
- **ADWIN** — Detects changes in model error rate across batches.
- **Label drift** — Z-test on the observed fraud rate vs. training rate.

### CSV Upload

Upload any transaction CSV with at least `amount` and `is_fraud` columns. The system:
- Flexibly maps column names (e.g. `amt`, `transaction_amount`, `trans_amt` all work)
- Synthesizes missing features with reasonable distributions
- Works with the [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset out of the box

## Project Structure

```
├── api.py                 # FastAPI backend (upload, demo, metrics, retrain)
├── data_simulator.py      # Transaction generator with temporal drift
├── model.py               # LightGBM + XGBoost ensemble pipeline
├── drift_detector.py      # PSI, KS, ADWIN, label drift detectors
├── monitor.py             # Production monitoring simulation
├── run_pipeline.py        # CLI-only version (generates PNG plots)
├── start.sh               # One-command launcher
├── requirements.txt
└── frontend/
    └── src/
        ├── App.jsx
        └── components/
            ├── UploadPage.jsx        # CSV upload / demo data landing
            ├── Overview.jsx          # Stat cards
            ├── MetricsChart.jsx      # AUPRC/precision/recall over time
            ├── DriftChart.jsx        # PSI timeline with thresholds
            ├── AlertsPanel.jsx       # Filterable alert list
            ├── FraudMixChart.jsx     # Fraud type composition
            ├── FeatureImportance.jsx # Ensemble feature rankings
            ├── TransactionTable.jsx  # Scored transaction stream
            └── RetrainPanel.jsx      # Retrain with before/after
```

## Tech Stack

**ML:** Python, scikit-learn, LightGBM, XGBoost, imbalanced-learn, SciPy

**Backend:** FastAPI, uvicorn

**Frontend:** React, Recharts, Vite
