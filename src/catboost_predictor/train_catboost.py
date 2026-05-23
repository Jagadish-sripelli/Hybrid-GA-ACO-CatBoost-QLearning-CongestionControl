"""
train_catboost.py
-----------------
CatBoost congestion prediction model training.
Implements Section 5.3 of the paper.

Congestion classes:
    0 — Normal   (P_cong < 0.3)
    1 — Moderate (0.3 ≤ P_cong < 0.7)
    2 — Critical (P_cong ≥ 0.7)
"""

import numpy as np
import pandas as pd
import os
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler


# ── Feature columns (18 features as described in paper Section 7.3) ──────────
FEATURE_COLS = [
    'bandwidth_util',      # current bandwidth utilization (%)
    'queue_occupancy',     # queue occupancy ratio [0-1]
    'end_to_end_delay',    # ms
    'packet_loss_rate',    # [0-1]
    'jitter',              # ms
    'hop_count',           # integer
    'energy_consumption',  # J
    'mean_bandwidth',      # temporal mean
    'var_bandwidth',       # temporal variance
    'mean_delay',          # temporal mean
    'var_delay',           # temporal variance
    'mean_loss',           # temporal mean
    'var_loss',            # temporal variance
    'trend_bandwidth',     # trend indicator [-1, 0, 1]
    'trend_delay',
    'trend_queue',
    'link_quality_index',  # composite link quality score
    'flow_count',          # active flows on path
]

LABEL_COL = 'congestion_class'


def label_congestion(p_cong: float) -> int:
    """Map congestion probability to class label (Section 5.3)."""
    if p_cong < 0.3:
        return 0   # Normal
    elif p_cong < 0.7:
        return 1   # Moderate
    else:
        return 2   # Critical


def generate_synthetic_dataset(n_samples: int = 10000,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training dataset from simulated network states.
    Used when CAIDA/MAWI traces are not available.
    """
    rng = np.random.RandomState(seed)

    data = {
        'bandwidth_util':    rng.uniform(0, 100, n_samples),
        'queue_occupancy':   rng.uniform(0, 1,   n_samples),
        'end_to_end_delay':  rng.uniform(1, 200, n_samples),
        'packet_loss_rate':  rng.uniform(0, 0.3, n_samples),
        'jitter':            rng.uniform(0, 50,  n_samples),
        'hop_count':         rng.randint(1, 15,  n_samples),
        'energy_consumption':rng.uniform(0.01, 5.0, n_samples),
        'mean_bandwidth':    rng.uniform(0, 100, n_samples),
        'var_bandwidth':     rng.uniform(0, 500, n_samples),
        'mean_delay':        rng.uniform(1, 200, n_samples),
        'var_delay':         rng.uniform(0, 100, n_samples),
        'mean_loss':         rng.uniform(0, 0.3, n_samples),
        'var_loss':          rng.uniform(0, 0.1, n_samples),
        'trend_bandwidth':   rng.choice([-1, 0, 1], n_samples),
        'trend_delay':       rng.choice([-1, 0, 1], n_samples),
        'trend_queue':       rng.choice([-1, 0, 1], n_samples),
        'link_quality_index':rng.uniform(0, 1,   n_samples),
        'flow_count':        rng.randint(1, 50,  n_samples),
    }

    df = pd.DataFrame(data)

    # Compute congestion probability as function of queue + loss + delay
    p_cong = (
        0.4 * df['queue_occupancy'] +
        0.3 * df['packet_loss_rate'] / 0.3 +
        0.2 * df['end_to_end_delay'] / 200 +
        0.1 * df['bandwidth_util'] / 100
    )
    p_cong = np.clip(p_cong + rng.normal(0, 0.05, n_samples), 0, 1)
    df[LABEL_COL] = p_cong.apply(label_congestion)

    return df


def train_model(data_path: str = None,
                model_save_path: str = 'results/models/catboost_model.pkl',
                n_samples: int = 10000,
                seed: int = 42,
                verbose: bool = True) -> CatBoostClassifier:
    """
    Train CatBoost congestion prediction model.

    Parameters
    ----------
    data_path : str or None
        Path to CSV dataset. If None, generates synthetic data.
    model_save_path : str
        Where to save the trained model
    n_samples : int
        Number of synthetic samples if data_path is None
    seed : int
    verbose : bool

    Returns
    -------
    Trained CatBoostClassifier
    """
    # ── Load or generate data ─────────────────────────────────────────────────
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path}: {len(df)} samples")
    else:
        print(f"Generating synthetic dataset ({n_samples} samples)...")
        df = generate_synthetic_dataset(n_samples, seed)

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    # ── Normalize features ────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # ── Train/validation split (75:25 per paper Section 5.3) ─────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # ── Train CatBoost ────────────────────────────────────────────────────────
    model = CatBoostClassifier(
        iterations=500,            # num_trees per paper Table 3
        learning_rate=0.05,        # per paper Table 3
        depth=8,                   # max tree depth per paper Table 3
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=seed,
        verbose=100 if verbose else 0,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred,
                                 target_names=['Normal', 'Moderate', 'Critical']))

    # ── Save model and scaler ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"\nModel saved to {model_save_path}")

    return model, scaler


if __name__ == '__main__':
    train_model(verbose=True)
