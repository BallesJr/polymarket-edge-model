import pandas as pd
import numpy as np
import requests
import json
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from polymarket_api import fetch_all_active_markets

# URLs API bases
GAMMA_URL = "https://gamma-api.polymarket.com" # Gamma API: used to download resolved (closed) markets for training
CLOB_URL  = "https://clob.polymarket.com"

# Downloads closed markets from 2024
def fetch_resolved_markets(max_markets: int = 3000) -> pd.DataFrame:
    all_markets = []
    offset = 0
    batch = 100

    print("Downloading resolved markets...")

    while len(all_markets) < max_markets:
        params = {
            "limit": batch,
            "offset": offset,
            "active": "false",
            "closed": "true",
            "end_date_min": "2024-01-01", # Older markets lack reliable price data
        }

        try:
            resp = requests.get(f"{GAMMA_URL}/markets", params=params, timeout=10)
            resp.raise_for_status()
            batch_data = resp.json()
        except requests.RequestException as e:
            print(f"\nAPI error: {e}")
            break
        
        if not batch_data:
            break 
        
        remaining = max_markets - len(all_markets)
        all_markets.extend(batch_data[:remaining]) # Never exceed max_markets
        offset += len(batch_data)
        print(f" -> {len(all_markets)} resolved markets", end="\r")

        if len(batch_data) < batch:
            break # Last page

        time.sleep(0.3)

    print(f"\nTotal: {len(all_markets)} resolved markets found.")
    return _parse_resolved_markets(all_markets)
    
# Extracts outcomes (YES = 1/NO = 0) and pre-resolution probability from each resolved market
def _parse_resolved_markets(markets: list[dict]) -> pd.DataFrame:
    rows = []

    for m in markets:
        try:
            prices = json.loads(m.get("outcomePrices", "[]"))
            if not prices or len(prices) < 2:
                continue
            p0, p1 = float(prices[0]), float(prices[1])
        except (json.JSONDecodeError, ValueError):
            continue
        
        # Skip markets with ambiguous resolution (e.g. ["0.5", "0.5"])
        if p0 == 1.0 and p1 == 0.0:
            outcome = 1 # YES won
        elif p0 == 0.0 and p1 == 1.0:
            outcome = 0 # NO won
        else:
            continue

        # Uses (bestBid + bestAsk) / 2 as the pre-resolution probability
        bid = m.get("bestBid")
        ask = m.get("bestAsk")
        if bid is None or ask is None:
            continue

        try:
            prob_market = (float(bid) + float(ask)) / 2
        except (TypeError, ValueError):
            continue

        # Skip markets where the price was already at the extremes before resolution
        if not (0.01 <= prob_market <= 0.99):
            continue
        
        # Calculates the market duration: longer markets may have different calibration patterns
        try:
            start = pd.to_datetime(m.get("startDate"), utc=True, errors="coerce")
            end = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")
            duration_days = (end - start).days if pd.notna(start) and pd.notna(end) else None
        except Exception:
            duration_days = None

        rows.append({
            "outcome": outcome, # Ground truth: 1=YES won, 0=NO won
            "prob_market": prob_market, # Market's implied probability before resolution
            "spread": m.get("spread"),
            "liquidity": m.get("liquidityNum") or m.get("liquidity"),
            "volume_total": m.get("volumeNum") or m.get("volume"),
            "competitive": m.get("competitive"),
            "duration_days": duration_days,
            "change_1d": m.get("oneDayPriceChange"),
            "change_1w": m.get("oneWeekPriceChange"),
            "market_id": m.get("id"),
            "question": m.get("question"),
            "end_date": m.get("endDate"),
        })

    df = pd.DataFrame(rows)

    # Forces numeric types (some fields com as strings)
    numeric_cols = ["prob_market", "spread", "liquidity", "volume_total", "competitive", "duration_days", "change_1d", "change_1w"]
    for col in numeric_cols:
        if col in df.columns:
           df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Clean resolved markets: {len(df)}")

    return df

# Transforms raw columns into model-ready features (handles missing values and skewed distributions)
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    # Log transforms compresses large ranges
    # Uses clip(lower=1e-10) to avoid log(0) which would produce -inf
    df["log_liquidity"] = np.log1p(
        pd.to_numeric(df["liquidity"], errors="coerce").fillna(0).clip(lower=1e-10)
    )
    df["log_volume"] = np.log1p(
        pd.to_numeric(df["volume_total"], errors="coerce").fillna(0).clip(lower=1e-10)
    )
    df["spread_clean"] = df["spread"].fillna(df["spread"].median()) # Fill with median, not 0
    df["competitive_clean"] = df["competitive"].fillna(0.5) # 0.5 = neutral when unknown
    df["change_1d_clean"] = df["change_1d"].fillna(0) # No change when data missing
    df["change_1w_clean"] = df["change_1w"].fillna(0)
    df["abs_change_1d"] = df["change_1d_clean"].abs() # Magnitud matters, not direction
    df["abs_change_1w"] = df["change_1w_clean"].abs()

    # Create duration_days column if missing (active markets don't have it)
    if "duration_days" not in df.columns:
        df["duration_days"] = None
    df["duration_clean"] = np.log1p(pd.to_numeric(df["duration_days"], errors="coerce").fillna(30).clip(lower=1e-10)).clip(lower=0)

    # Distance to 50% captures how uncertain market is
    df["dist_to_50"] = (df["prob_market"] - 0.5).abs()
    # Longshot bias: markets with very low probabilities tend to be overpriced
    df["is_longshot"] = (df["prob_market"] < 0.15).astype(int)

    feature_cols = ["prob_market", "log_liquidity", "log_volume", "spread_clean", "competitive_clean", "duration_clean", "change_1d_clean", "change_1w_clean", "abs_change_1d", "abs_change_1w", "dist_to_50", "is_longshot"]

    # Replaces any remaining value with NaN so dropna() handles them cleanly
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    return df, feature_cols

# Trains Logistic Regression and Random Forest - returns the best one by cross-validated Brier Score
def train_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    df_clean = df[feature_cols + ["outcome"]].dropna()
    print(f"\nTraining set: {len(df_clean)} markets (after dropping Nan)")
    if len(df_clean) < 50:
        print("WARNING: There isn't enough markets to train. You need at least 50.")
        return None, None, None, feature_cols
    
    X = df_clean[feature_cols].copy()
    y = df_clean["outcome"].values # 1 = YES won, 0 = NO won
    # Model 1: Logisitic regression
    # Pipeline = StandardScaler + model - scaler ensure features are on the same scale
    lr_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")) # class-weight="balanced" compensates for the 3:1 NO/YES imbalance in the dataset
        ])
    #cross_val_score with cv=5: splits data into 5 folds, trains on 4 and validates on 1
    lr_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring="neg_brier_score") # neg_brier_score: Brier Score measures probability calibration (lower = better)
    lr_brier = -lr_scores.mean()
    # Model 2: Random forest
    rf_pipeline =  Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=5, class_weight="balanced"))
    ])
    rf_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring="neg_brier_score")
    rf_brier = -rf_scores.mean()

    print(f"\nModel comparison(Brier Score - lower is better):")
    print(f"Logistic Regression: {lr_brier:.4f} ± {lr_scores.std():.4f}")
    print(f"Random forest: {rf_brier:.4f} ± {rf_scores.std():.4f}")

    # baseline: Brier Score if we just used the raw market probability as our prediction
    market_brier = brier_score_loss(y, df_clean["prob_market"].values)
    print(f"Market(baseline): {market_brier:.4f}") 

    # Picks the model with the lower Brier Score and retrain on the full dataset (decides which model is better)
    if lr_brier <= rf_brier:
        best_pipeline = lr_pipeline
        best_name = "Logistic Regressions"
        best_brier = lr_brier
    else:
        best_pipeline = rf_pipeline
        best_name = "Random Forest"
        best_brier = rf_brier
    
    best_pipeline.fit(X, y)
    print(f"\nBest model: {best_name} (Brier: {best_brier:.4f})")

    return best_pipeline, X, y, feature_cols

# Generates two plots: Reliability Diagram and Brier Score comparison bar chart
def plot_calibration(model, X: np.ndarray, y: np.ndarray, prob_market: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0])

    # calibration_curve bins predictions and computes actual outcome rates per bin
    frac_pos_mkt, mean_pred_mkt = calibration_curve(y, prob_market, n_bins=10, strategy="uniform")
    prob_model_pred = model.predict_proba(X)[:, 1]
    frac_pos_mod, mean_pred_mod = calibration_curve(y, prob_model_pred, n_bins=10, strategy="uniform")

    ax1.plot([0,1], [0, 1], "k--", alpha=0.5, label="Perfect calibration") # Diagonal = Perfect
    ax1.plot(mean_pred_mkt, frac_pos_mkt, "o-", color="#1f77b4", label="Polymarket (raw)", linewidth=2)

    ax1.plot(mean_pred_mod, frac_pos_mod, "s-", color="#d62728", label="Our model", linewidth=2)
    ax1.set_xlabel("Mean predicted probability", fontsize=12)
    ax1.set_ylabel("Fraction of positives", fontsize=12)
    ax1.set_title("Reliability Diagram", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[1])

    brier_market = brier_score_loss(y, prob_market)
    brier_model = brier_score_loss(y, prob_model_pred)
    brier_naive = brier_score_loss(y, np.full_like(y, 0.5, dtype=float)) # Always predict 50%

    models = ["Naive\n(always 0.5)", "Polymarket\n(raw)", "Our model"]
    scores = [brier_naive, brier_market, brier_model]
    colors = ["#aec7e8", "#1f77b4", "#d62728"]

    bars = ax2.bar(models, scores, color=colors, alpha=0.85, width=0.5)

    for bar, score in zip(bars, scores):
        ax2.text(
            bar.get_x() +  bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax2.set_ylabel("Brier Score (lower = better)", fontsize=12)
    ax2.set_title("Brier Score Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(scores) * 1.2)
    ax2.grid(True, alpha=0.3, axis="y")

    improvement = (brier_market - brier_model) / brier_market * 100
    fig.suptitle(f"Polymarket Calibration Analysis |"
                 f"Model improvement: {improvement:+.1f}% over raw market",
                 fontsize=13, y=1.02)
    
    plt.tight_layout()
    plt.savefig("data/calibration_analysis.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: data/calibration_analysis.png")
    plt.show()

# Applies the trained model to active markets and returns actionable signals
def predict_active_markets(model, df_active: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    from expected_value import expected_value, kelly_fraction, filter_tradeable_markets

    df_tradeable = filter_tradeable_markets(df_active)
    if df_tradeable.empty:
        print("No tradeable active markets found.")
        return pd.DataFrame()
    
    # Active markets use "prob_yes_market", so it renames to match training column name
    if "prob_yes_market" in df_tradeable.columns and "prob_market" not in df_tradeable.columns:
        df_tradeable = df_tradeable.rename(columns={"prob_yes_market": "prob_market"})

    # Active markets may lack some columns present in resolved markets - fill with None
    for col in ["change_1d", "change_1w", "competitive", "liquidity", "volume_total"]:
        if col not in df_tradeable.columns:
            df_tradeable[col] = None

    df_feat, _ = build_features(df_tradeable)

    df_feat_clean = df_feat.dropna(subset=feature_cols)

    if df_feat_clean.empty:
        print("No market with complete features.")
        return pd.DataFrame()
    
    X_active = df_feat_clean[feature_cols]

    prob_model = model.predict_proba(X_active)[:, 1] # Predicted P(YES wins)

    signals = []
    for idx, (_, row) in enumerate(df_feat_clean.iterrows()):
        # Uses the original market name for prob
        prob_mkt = row["prob_yes_market"] if "prob_yes_market" in row else row["prob_market"]
        prob_mod = prob_model[idx]
        spread = row["spread"] if pd.notna(row.get("spread")) else 0.02

        ev = expected_value(prob_mod, prob_mkt)
        kelly = kelly_fraction(prob_mod, prob_mkt)

        # Only reports signals where edge exceeds transaction cost + minimum threshold
        if abs(ev) < spread + 0.05:
            continue

        signals.append({
            "market_id": row.get("market_id"),
            "question": row.get("question"),
            "end_date": row.get("end_date"),
            "prob_market": round(prob_mkt, 4),
            "prob_model": round(prob_mod, 4),
            "edge": ev,
            "kelly_fraction": kelly,
            "direction": "BUY YES" if ev > 0 else "BUY NO",
            "spread": spread,
            "liquidity": row.get("liquidity"),
            "source": "calibration_model",
        })

    if not signals:
        print("No signals above thresold.")
        return pd.DataFrame()
        
    # Sort by absolute edge (largest opportunities first)
    return pd.DataFrame(signals) \
            .sort_values("edge", key=abs, ascending=False) \
            .reset_index(drop=True)

# Main
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("=== Polymarket Calibration Model ===")
    # Step 1: Downloads resolved markets (training data)
    df_resolved = fetch_resolved_markets(max_markets=3000)

    if df_resolved.empty:
        print("No resolved markets founds. Exiting.")
        exit()
    # Step 2: Builds features
    df_feat, feature_cols = build_features(df_resolved)
    print(f"\nFeatures: {feature_cols}")
    print(f"Class balance: {df_feat["outcome"].value_counts().to_dict()}")
    # Step 3: Trains and compares the models
    model, X, y, feature_cols = train_model(df_feat, feature_cols)

    if model is None:
        print("Training failed. Exiting.")
        exit()
    # Step 4: Reliability Diagram + Brier Score
    prob_market_arr = df_feat.dropna(subset=feature_cols)["prob_market"].values
    plot_calibration(model, X, y, prob_market_arr)
    # Step 5: Applies model to active markets and print signals
    print("\nLoading active markets...")
    df_active = fetch_all_active_markets(max_markets=300)

    df_signals = predict_active_markets(model, df_active, feature_cols)

    if not df_signals.empty:
        print(f"\nSIGNALS FROM CALIBRATION MODEL: {len(df_signals)}")
        cols = ["question", "prob_market", "edge", "kelly_fraction", "direction", "spread"]
        print(df_signals[cols].to_string(index=False))

        print("\nNOTE: Signals are model-based estimates. Always verify manually before trading.")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        path = f"data/calibration_signals_{ts}.csv"
        df_signals.to_csv(path, index=False)
        print(f"\nSaved to: {path}")

# PRODUCTION MODEL (Canonical Random Forest trainer)
# train_model() above is a research function: it compares Logistic Regression vs Random Forest and picks the winner
# That comparison showed RF wins consistently on Polymarket data
# So I promote it here as the single canonical production model

# Both backtester.py and signal_engine.py import this function
# This function guarantees by construction that we backtest is what runs live
# If I ever want to change hyperparameters, I change them once here

# Trains the canonical production Random Forest on the provided dataset
# This is the single source of truth for the model used across the projecte:
# - backtester.py calls this to validate historical performance
# - signal_engine.py calls this to generate live trading signals
# Keeping both callers on the same function means the backtest results are a faithful estimate of live performance
def train_random_forest(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    df_clean = df[feature_cols + ["outcome"]].dropna()

    if len(df_clean) < 50:
        raise ValueError(
            f"Only {len(df_clean)} clean samples available after dropping Nan."
            f"Need at least 50 to train a reliable model."
        ) 
    
    X = df_clean[feature_cols]
    y = df_clean["outcome"].values # 1 = YES won, 0 = NO won

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100, # Stable predictions without excessive training time
            min_samples_leaf=5, # Prevent overfitting on small market subsets
            class_weight="balanced", # Corrects the NO/YES imbalance in resolved Polymarket markets
            random_state=42, # Fully reproducible results across runs
        ))
    ])

    pipeline.fit(X, y)

    brier = brier_score_loss(y, pipeline.predict_proba(X)[:, 1])
    print(f"[RF] Trained on {len(df_clean)} markets | Brier (train): {brier:.4f}")

    return pipeline