import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Reuse functions from our project
from calibration import fetch_resolved_markets, build_features, train_random_forest
from expected_value import expected_value, kelly_fraction

# BACKTESTER - Historical P&L Simulation
# Goal: I want to see if I had traded with our model the past months, how much would we have made or lost?
# Key: I use a temporal split (not random) to avoid look-ahead bias

# Splits dataset into traind and test by chronological order
# With a random split, the model could trains on some markets and test on previous markets (it sees the future)
# If it could happen, it's called look-ahead bias
# With a temporal split, the model trains on the first 70% of markets (by date) and test on the last 30%
# Exactly as we would in production
def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["end_date"]).sort_values("end_date").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"\n--- Temporal Split ---")
    print(f"Train: {len(df_train)} markets ({df_train["end_date"].min().date()} -> {df_train['end_date'].max().date()})")
    print(f"Test: {len(df_test)} markets ({df_test["end_date"].min().date()} -> {df_test['end_date'].max().date()})")

    return df_train, df_test

# Trains the Random Forest on the training split
def train_on_split(df_train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
   return train_random_forest(df_train, feature_cols)

# Simulates trades on the test set
# For each test market:
# 1. Model predicts P(YES) (prob_model)
# 2. Compare with market price (edge = prob_model - prob_market)
# 3. If |edge| > min_edge + spread, trades
# 4. Positions size via Kelly (capped at max_kelly)
# 5. Payoff depends on the actual outcome 
def simulate_trades(
        model: Pipeline,
        df_test: pd.DataFrame,
        feature_cols: list[str],
        min_edge: float = 0.02, # Minimum edge to trigger a trade
        max_kelly: float = 0.10, # Cap to avoid oversized positions
        max_position: float = 500.0,
        bankroll: float = 10_000.0, # Starting capital in $
) -> pd.DataFrame:
    
    df_feat, _ = build_features(df_test)
    df_clean = df_feat.dropna(subset=feature_cols + ["outcome"]).copy()

    if df_clean.empty:
        print("No calid markets in the test set.")
        return pd.DataFrame()

    X_test = df_clean[feature_cols]
    prob_model_arr = model.predict_proba(X_test)[:, 1]

    # Edge diagnostics: shows the distribution before filtering
    # Resolved markets have unrealiable spreads (market is closed, no one is trading)
    # I cap at 0.05 (5%) which is a realistic spread for an active Polymarket market
    SPREAD_CAP = 0.05
    all_edges = prob_model_arr - df_clean["prob_market"].values
    all_spreads = df_clean["spread_clean"].fillna(0.02).clip(upper=SPREAD_CAP).values
    all_thresholds = all_spreads + min_edge
    tradeable_mask = np.abs(all_edges) > all_thresholds

    print(f"\nEdge diagnostics ({len(df_clean)} test markets):")
    print(f"Min edge: {np.min(np.abs(all_edges)):.4f} | Max edge: {np.max(np.abs(all_edges)):.4f}")
    print(f"Mean |edge|: {np.mean(np.abs(all_edges)):.4f} | Median |edge|: {np.median(np.abs(all_edges)):.4f}")
    print(f"Threshold (min_edge + spread): ~{np.mean(all_thresholds):.4f}")
    print(f"Markets passing filter: {tradeable_mask.sum()} / {len(df_clean)}")

    trades = []
    current_bankroll = bankroll
    skipped_kelly = 0 # Track how many are killed by Kelly returning 0

    for idx, (i, row) in enumerate(df_clean.iterrows()):
        prob_market = row["prob_market"]
        prob_model = prob_model_arr[idx]
        spread = min(row.get("spread_clean", 0.02) if pd.notna(row.get("spread_clean")) else 0.02, SPREAD_CAP)
        outcome = row["outcome"] # 1 = YES won, 0 = NO won

        edge = expected_value(prob_model, prob_market)

        # Filter: edge must exceed spread + minimum threshold
        if abs(edge) < spread + min_edge:
            continue

        direction = "BUY YES" if edge > 0 else "BUY NO"

        # Position sizing
        
        kelly = kelly_fraction(prob_model, prob_market, half_kelly=True)
        kelly = min(kelly, max_kelly) # Cap for safety, to avoid oversized bets
        position_size = current_bankroll * kelly

        # Liquidity cap: can't bet more than a fraction of the market's actual liquidity
        # I use 10% of liquidity (taking more would move the price against us: slippage)
        market_liquidity = row.get("liquidity", max_position)
        if pd.isna(market_liquidity) or market_liquidity <= 0:
            market_liquidity = max_position
        liquidity_cap = float (market_liquidity) * 0.10
        position_size = min(position_size, liquidity_cap, max_position) 

        if position_size < 1.0: 
            continue # Position too small, skip

        # P&L calculations based on whether we got it right
        if direction == "BUY YES":
            price_paid = prob_market # Purchase price for YES token
            if outcome == 1: # YES wins (collect 1$ per token)
                pnl = position_size * (1 - price_paid) / price_paid
            else: # NO wins (lose entire position)
                pnl = -position_size
        else: # BUY NO
            price_paid = 1 - prob_market # Purchase price for NO token
            if outcome == 0: # NO wins (collect 1$ per token)
                pnl = position_size * (1 - price_paid) / price_paid
            else: # YES wins (lose entire position)
                pnl = -position_size

        current_bankroll += pnl

        trades.append({
            "date": row.get("end_date"),
            "question": row.get("question", "")[:80],
            "direction": direction,
            "prob_market": round(prob_market, 4),
            "edge": edge,
            "kelly": round(kelly, 4),
            "position_size": round(position_size, 2),
            "outcome": outcome,
            "won": int((direction == "BUY YES" and outcome == 1) or
                       (direction == "BUY NO" and outcome == 0)),
            "pnl": round(pnl, 2),
            "bankroll": round(current_bankroll, 2),
        })

    if skipped_kelly > 0: 
            print(f"Skipped by Kelly (position < 1$): {skipped_kelly}")

    return pd.DataFrame(trades)
    
# Computes key backtest metrics
# Win rates: % of profitable trades
# ROI: return of investment (P&L / initial capital)
# Max drawdown: largest peak-to-through decline (worst moment)
# Profit factor: gross profits / gross losses (>1 = prfitable)
def compute_metrics(df_trades: pd.DataFrame, initial_bankroll: float = 10_000.0) -> dict:
    if df_trades.empty:
        return {}
    
    total_pnl = df_trades["pnl"].sum()
    wins = df_trades["won"].sum()
    total = len(df_trades)

    # Max drawdown
    bankroll_series = df_trades["bankroll"].values
    running_max = np.maximum.accumulate(bankroll_series)
    drawdowns = (bankroll_series - running_max) / running_max
    max_dd = drawdowns.min()

    # Profit factor
    gross_profit = df_trades.loc[df_trades["pnl"] > 0, "pnl"]. sum()
    gross_loss = abs(df_trades.loc[df_trades["pnl"] < 0, "pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return{
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl": round(total_pnl / initial_bankroll * 100, 2),
        "roi_pct": round(total_pnl / initial_bankroll * 100, 2),
        "final_bankroll": round(initial_bankroll + total_pnl, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_edge": round(df_trades["edge"].mean(), 4),
        "avg_position": round(df_trades["position_size"].mean(), 2),
    }

# Generates 3 plots:
# 1. Equity curve (bankroll evolution over time)
# 2. P&L per trade (green/red bars)
# 3. Edge distribution of executed trades
def plot_backtest(df_trades: pd.DataFrame, metrics: dict, initial_bankroll: float = 10_000.0) -> None:
    if df_trades.empty:
        print("No trades to plot.")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0,:]) # Full top row
    ax1.plot(df_trades["bankroll"].values, color="#1f77b4", linewidth=2, label="Bankroll")
    ax1.axhline(initial_bankroll, color="gray", linestyle="--", alpha=0.5, label="Initial capital")
    ax1.fill_between(range(len(df_trades)),
                     df_trades["bankroll"].values,
                     initial_bankroll,
                     where=df_trades["bankroll"].values >= initial_bankroll,
                     color="#2ca02c", alpha=0.15)
    ax1.fill_between(range(len(df_trades)),
                     df_trades["bankroll"].values,
                     initial_bankroll,
                     where=df_trades["bankroll"].values < initial_bankroll,
                     color="#d62728", alpha=0.15)
    ax1.set_title(
        f"Equity Curve | ROI: {metrics["roi_pct"]}% | Max Drawdown: {metrics["max_drawdown_pct"]}%",
        fontsize=13, fontweight="bold"
    )
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Bankroll ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. P&L per trade
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ["#2ca02c" if p > 0 else "#d62728" for p in df_trades["pnl"]]
    ax2.bar(range(len(df_trades)), df_trades["pnl"].values, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title(f"P&L per Trade | Win Rate: {metrics["win_rate"]}%",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("P&L ($)")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Edge distribution
    ax3 = fig.add_subplot(gs[1,1])
    ax3.hist(df_trades["edge"].values, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax3.axvline(df_trades["edge"].mean(), color="#d62728", linestyle="--", label=f"Avg edge: {metrics["avg_edge"]:.4f}")
    ax3.set_title("Edge Distribution", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Edge (prob_model - prob_market)")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Backtest Polymarket Edge Model | {metrics["total_trades"]} trades | "
        f"PF: {metrics["profit_factor"]}",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig("data/backtest_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to. data/backtest_results.png")
    plt.show()

# Print backtest metrics in a clean format
def print_metrics(metrics: dict) -> None:
    print(f"\n{"="*50}")
    print(f"BACKTEST RESULTS")
    print(f"{"="*50}")
    print(f"Total trades: {metrics["total_trades"]}")
    print(f"Wins: {metrics["wins"]} ({metrics["win_rate"]}%)")
    print(f"Losses: {metrics["losses"]}")
    print(f"{chr(9472)*46}")
    print(f"Total P&L: ${metrics["total_pnl"]:+.2f}")
    print(f"ROI: {metrics["roi_pct"]:+.2f}%")
    print(f"Final bankroll: ${metrics["final_bankroll"]:.2f}")
    print(f"{chr(9472)*46}")
    print(f"Max Drawdown: {metrics["max_drawdown_pct"]:.2f}%")
    print(f"Profit Factor: {metrics["profit_factor"]}")
    print(f"Avg position: ${metrics["avg_position"]:.2f}")
    print(f"{"="*50}")

# MAIN
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    INITIAL_BANKROLL = 10_000.0
    MIN_EDGE = 0.02 # Minimum edge to trade (2%)
    USE_KELLY = True # True = Half-Kelly, False = fixed 2% fraction
    MAX_KELLY = 0.10 # Max 10% of bankroll per trade
    MAX_POSITION = 500.0 # Max $500 per trade (realistic Polymarket liquidity)
    TRAIN_RATIO = 0.70 # 70% train, 30% test

    print("="*50)
    print("POLYMARKET BACKTESTER")
    print("="*50)

    # STEP 1: Download resolved markets
    print("\n[1/5] Downloading resolved markets")
    df_resolved = fetch_resolved_markets(max_markets=3000)
    if df_resolved.empty:
        print("No markets found. Exiting.")
        exit()
    
    # STEP 2: Build features
    print("\n[2/5] Building features...")
    df_feat, feature_cols = build_features(df_resolved)
    print(f"Features: {feature_cols}")
    print(f"Class balance: {df_feat["outcome"].value_counts().to_dict()}")

    # STEP 3: Temporal split
    print("\n[3/5] Temporal train/tes split...")
    df_train, df_test = temporal_split(df_feat, train_ratio=TRAIN_RATIO)

    # STEP 4: Train model only on train set
    print("\n[4/5] Training model...")
    model = train_on_split(df_train, feature_cols)

    # Out-of-sample Brier Score on test
    df_test_clean = df_test.dropna(subset=feature_cols + ["outcome"])
    if len(df_test_clean) > 0:
        X_test = df_test_clean[feature_cols]
        y_test = df_test_clean["outcome"].values
        prob_test = model.predict_proba(X_test)[:, 1]
        brier_test = brier_score_loss(y_test, prob_test)
        brier_market = brier_score_loss(y_test, df_test_clean["prob_market"].values)
        print(f"Brier Score test (model): {brier_test:.4f}")
        print(f"Brier Score test (market): {brier_market:.4f}")

        improvement = (brier_market - brier_test) / brier_market * 100
        print(f"Improvement over market: {improvement:+.1f}%")

    # STEP 5: Simulate trades
    print("\n[5/5] Simulating trades...")
    df_trades = simulate_trades (
        model=model,
        df_test=df_test,
        feature_cols=feature_cols,
        min_edge=MIN_EDGE,
        max_kelly=MAX_KELLY,
        max_position=MAX_POSITION,
        bankroll=INITIAL_BANKROLL
    )

    if df_trades.empty:
        print("No trades generated. Try reducing min_edge.")
        exit()

    # Metrics and visualization
    metrics = compute_metrics(df_trades, initial_bankroll=INITIAL_BANKROLL)
    print_metrics(metrics)

    print("TOP 5 best trades:")
    print(df_trades.nlargest(5, "pnl")[["question", "direction", "edge", "pnl", "won"]].to_string(index=False))
    print("\n TOP 5 worst trades:")
    print(df_trades.nsmallest(5, "pnl")[["question", "direction", "edge", "pnl", "won"]].to_string(index=False))

    plot_backtest(df_trades, metrics, initial_bankroll=INITIAL_BANKROLL)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df_trades.to_csv(f"data/backtest_trades_{ts}.csv", index=False)
    print(f"Trades saved to: data/backtest_trades_{ts}.csv")

    