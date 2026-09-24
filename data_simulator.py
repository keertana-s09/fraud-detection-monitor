"""
Realistic transaction data simulator with concept drift.

Generates synthetic credit card transactions where fraud patterns evolve over time,
mimicking real-world behavior: fraudsters adapt, new attack vectors emerge, and
old ones get patched. Fraud rate is kept under 1% to reflect real class imbalance.

Fraud typologies and their drift behavior:
  - Card-not-present (CNP) fraud: starts common, declines as 3DS adoption increases
  - Account takeover (ATO): grows over time as phishing scales
  - Bust-out fraud: periodic bursts (organized rings)
  - Micro-charge testing: emerges mid-timeline (new attack vector)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DriftConfig:
    """Controls how fraud patterns shift across time periods."""
    # Period boundaries as fractions of total timeline [0, 1]
    # Period 0: baseline, Period 1: transition, Period 2: new regime
    period_boundaries: tuple = (0.0, 0.4, 0.7, 1.0)

    # Fraud rate per period (always < 1%)
    fraud_rates: tuple = (0.005, 0.006, 0.008)

    # Fraud type mix per period: (cnp, ato, bust_out, micro_test)
    # Each tuple sums to 1.0
    fraud_mix_by_period: tuple = (
        (0.60, 0.25, 0.15, 0.00),  # Period 0: CNP dominant, no micro-testing yet
        (0.35, 0.35, 0.15, 0.15),  # Period 1: ATO rises, micro-testing appears
        (0.15, 0.45, 0.10, 0.30),  # Period 2: ATO dominant, micro-testing common
    )

    # Feature distribution shifts per period (mean shift for amount, hour, etc.)
    amount_mean_shift: tuple = (0.0, 50.0, 120.0)  # fraudsters go bigger over time
    hour_concentration_shift: tuple = (0.0, 2.0, 4.0)  # shift peak fraud hour


class TransactionSimulator:
    """Generates realistic transaction streams with evolving fraud patterns."""

    def __init__(self, seed: int = 42, drift_config: DriftConfig = None):
        self.rng = np.random.default_rng(seed)
        self.drift = drift_config or DriftConfig()

        # Merchant category codes and their legitimate transaction profiles
        self.mccs = {
            "grocery": {"mean_amt": 45, "std_amt": 30, "freq_weight": 0.30},
            "gas_station": {"mean_amt": 40, "std_amt": 15, "freq_weight": 0.15},
            "online_retail": {"mean_amt": 65, "std_amt": 50, "freq_weight": 0.20},
            "restaurant": {"mean_amt": 30, "std_amt": 20, "freq_weight": 0.15},
            "travel": {"mean_amt": 350, "std_amt": 250, "freq_weight": 0.05},
            "electronics": {"mean_amt": 200, "std_amt": 150, "freq_weight": 0.08},
            "subscription": {"mean_amt": 15, "std_amt": 8, "freq_weight": 0.07},
        }

    def _get_period(self, t_frac: float) -> int:
        """Determine which drift period a timestamp fraction falls into."""
        bounds = self.drift.period_boundaries
        for i in range(len(bounds) - 1):
            if bounds[i] <= t_frac < bounds[i + 1]:
                return i
        return len(bounds) - 2

    def _interpolate_param(self, t_frac: float, values: tuple) -> float:
        """Smoothly interpolate a parameter across periods (no hard jumps)."""
        period = self._get_period(t_frac)
        bounds = self.drift.period_boundaries

        # Position within current period [0, 1]
        period_start = bounds[period]
        period_end = bounds[period + 1]
        local_t = (t_frac - period_start) / (period_end - period_start)

        if period < len(values) - 1:
            # Smooth interpolation between current and next period
            return values[period] * (1 - local_t) + values[period + 1] * local_t
        return values[period]

    def _interpolate_mix(self, t_frac: float) -> np.ndarray:
        """Interpolate fraud type mixture weights across periods."""
        mixes = self.drift.fraud_mix_by_period
        period = self._get_period(t_frac)
        bounds = self.drift.period_boundaries

        period_start = bounds[period]
        period_end = bounds[period + 1]
        local_t = (t_frac - period_start) / (period_end - period_start)

        current = np.array(mixes[period])
        if period < len(mixes) - 1:
            next_mix = np.array(mixes[period + 1])
            blended = current * (1 - local_t) + next_mix * local_t
            return blended / blended.sum()
        return current

    def _generate_legit_transaction(self, customer_id: int) -> dict:
        """Generate a single legitimate transaction."""
        # Pick merchant category weighted by frequency
        cats = list(self.mccs.keys())
        weights = [self.mccs[c]["freq_weight"] for c in cats]
        mcc = self.rng.choice(cats, p=weights)
        profile = self.mccs[mcc]

        amount = max(0.50, self.rng.normal(profile["mean_amt"], profile["std_amt"]))
        hour = int(self.rng.normal(14, 4)) % 24  # peak around 2pm
        is_online = 1 if mcc in ("online_retail", "subscription") else int(self.rng.random() < 0.1)
        distance_from_home = max(0, self.rng.exponential(5))  # most transactions nearby

        return {
            "customer_id": customer_id,
            "amount": round(amount, 2),
            "merchant_category": mcc,
            "hour_of_day": hour,
            "is_online": is_online,
            "distance_from_home_km": round(distance_from_home, 2),
            "is_international": int(self.rng.random() < 0.03),
            "num_transactions_last_1h": self.rng.poisson(0.5),
            "num_transactions_last_24h": self.rng.poisson(3),
            "days_since_last_transaction": round(max(0, self.rng.exponential(1.5)), 1),
            "is_new_merchant": int(self.rng.random() < 0.15),
            "card_present": 1 - is_online,
            "is_fraud": 0,
            "fraud_type": "none",
        }

    def _generate_fraud_transaction(self, customer_id: int, fraud_type: str,
                                     t_frac: float) -> dict:
        """Generate a fraudulent transaction based on fraud typology."""
        amount_shift = self._interpolate_param(t_frac, self.drift.amount_mean_shift)
        hour_shift = self._interpolate_param(t_frac, self.drift.hour_concentration_shift)

        base = self._generate_legit_transaction(customer_id)
        base["is_fraud"] = 1
        base["fraud_type"] = fraud_type

        if fraud_type == "cnp":
            # Card-not-present: online, higher amounts, unusual hours
            base["is_online"] = 1
            base["card_present"] = 0
            base["amount"] = round(max(5, self.rng.lognormal(4.5, 1.0) + amount_shift), 2)
            base["hour_of_day"] = int(self.rng.normal(3 + hour_shift, 2)) % 24
            base["merchant_category"] = self.rng.choice(["online_retail", "electronics"])
            base["is_new_merchant"] = int(self.rng.random() < 0.7)
            base["distance_from_home_km"] = 0.0  # online

        elif fraud_type == "ato":
            # Account takeover: rapid sequence, unusual location, varied merchants
            base["amount"] = round(max(10, self.rng.lognormal(5.0, 0.8) + amount_shift), 2)
            base["hour_of_day"] = int(self.rng.normal(2 + hour_shift, 3)) % 24
            base["distance_from_home_km"] = round(self.rng.uniform(50, 500), 2)
            base["num_transactions_last_1h"] = self.rng.poisson(4)
            base["num_transactions_last_24h"] = self.rng.poisson(12)
            base["is_new_merchant"] = int(self.rng.random() < 0.85)
            base["is_international"] = int(self.rng.random() < 0.3)

        elif fraud_type == "bust_out":
            # Bust-out: very high amounts, travel/electronics, international
            base["amount"] = round(max(100, self.rng.lognormal(6.0, 0.5) + amount_shift), 2)
            base["merchant_category"] = self.rng.choice(["travel", "electronics"])
            base["is_international"] = int(self.rng.random() < 0.5)
            base["distance_from_home_km"] = round(self.rng.uniform(100, 2000), 2)
            base["num_transactions_last_24h"] = self.rng.poisson(8)

        elif fraud_type == "micro_test":
            # Micro-charge testing: tiny amounts, subscriptions, rapid fire
            base["amount"] = round(self.rng.uniform(0.50, 3.00), 2)
            base["merchant_category"] = self.rng.choice(["subscription", "online_retail"])
            base["is_online"] = 1
            base["card_present"] = 0
            base["num_transactions_last_1h"] = self.rng.poisson(6)
            base["is_new_merchant"] = int(self.rng.random() < 0.9)

        return base

    def generate(self, n_transactions: int = 500_000,
                 n_customers: int = 10_000,
                 start_date: str = "2024-01-01",
                 n_days: int = 365) -> pd.DataFrame:
        """
        Generate a full transaction dataset with temporal drift.

        Args:
            n_transactions: Total number of transactions to generate
            n_customers: Number of unique customers
            start_date: Start date for the timeline
            n_days: Number of days to simulate

        Returns:
            DataFrame sorted by timestamp with fraud labels and types
        """
        fraud_types = ["cnp", "ato", "bust_out", "micro_test"]
        records = []

        # Generate timestamps uniformly across the timeline
        timestamps = pd.date_range(start_date, periods=n_days, freq="D")
        txn_per_day = n_transactions // n_days

        for day_idx in range(n_days):
            t_frac = day_idx / n_days
            fraud_rate = self._interpolate_param(t_frac, self.drift.fraud_rates)
            fraud_mix = self._interpolate_mix(t_frac)

            # Add bust-out bursts: periodic spikes
            if fraud_types[2] == "bust_out":
                burst_phase = np.sin(2 * np.pi * t_frac * 6)  # ~6 bursts per year
                if burst_phase > 0.8:
                    fraud_mix[2] *= 3.0
                    fraud_mix = fraud_mix / fraud_mix.sum()

            n_daily = self.rng.poisson(txn_per_day)
            n_fraud_daily = self.rng.binomial(n_daily, fraud_rate)
            n_legit_daily = n_daily - n_fraud_daily

            date = timestamps[day_idx]

            # Generate legitimate transactions
            for _ in range(n_legit_daily):
                cust = self.rng.integers(0, n_customers)
                txn = self._generate_legit_transaction(cust)
                txn["timestamp"] = date + pd.Timedelta(
                    hours=int(txn["hour_of_day"]),
                    minutes=int(self.rng.integers(0, 60))
                )
                records.append(txn)

            # Generate fraud transactions
            for _ in range(n_fraud_daily):
                cust = self.rng.integers(0, n_customers)
                ftype = self.rng.choice(fraud_types, p=fraud_mix)
                txn = self._generate_fraud_transaction(cust, ftype, t_frac)
                txn["timestamp"] = date + pd.Timedelta(
                    hours=int(txn["hour_of_day"]),
                    minutes=int(self.rng.integers(0, 60))
                )
                records.append(txn)

        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        df["transaction_id"] = range(len(df))

        print(f"Generated {len(df):,} transactions over {n_days} days")
        print(f"  Fraud rate: {df['is_fraud'].mean():.4%}")
        print(f"  Fraud transactions: {df['is_fraud'].sum():,}")
        print(f"  Fraud type distribution:")
        fraud_counts = df[df["is_fraud"] == 1]["fraud_type"].value_counts()
        for ftype, count in fraud_counts.items():
            print(f"    {ftype}: {count:,} ({count / df['is_fraud'].sum():.1%})")

        return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that a real fraud team would engineer.
    These capture behavioral velocity, deviation from norms, and temporal patterns.
    """
    df = df.copy()

    # Amount-based features
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_zscore"] = (
        (df["amount"] - df.groupby("customer_id")["amount"].transform("mean"))
        / df.groupby("customer_id")["amount"].transform("std").clip(lower=1.0)
    )

    # Velocity features
    df["txn_velocity_ratio"] = (
        df["num_transactions_last_1h"] / df["num_transactions_last_24h"].clip(lower=1)
    )

    # Time-based features
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day

    # Interaction features
    df["online_night"] = df["is_online"] * df["is_night"]
    df["high_amount_new_merchant"] = (
        (df["amount"] > df["amount"].quantile(0.9)).astype(int) * df["is_new_merchant"]
    )
    df["international_high_amount"] = (
        df["is_international"] * (df["amount"] > 200).astype(int)
    )

    # Merchant category encoding (target encoding would leak — use frequency)
    mcc_freq = df["merchant_category"].value_counts(normalize=True)
    df["mcc_frequency"] = df["merchant_category"].map(mcc_freq)

    return df


if __name__ == "__main__":
    sim = TransactionSimulator(seed=42)
    df = sim.generate(n_transactions=200_000, n_days=365)
    df = add_engineered_features(df)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
