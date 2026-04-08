import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline

# I always use the Random Forest, the same model validated in backtester.py
# calibration.train_model() picks between RF and LR each run depending on the data
# This choice would make live performance inconsistent with the backtest results:
# train_random_forest() from calibration.py is the canonical RF definition
# backtester.py also uses it, guaranteeing backtest and live model are identical
from polymarket_api import fetch_all_active_markets
from calibration import fetch_resolved_markets, build_features, train_random_forest
from expected_value import filter_tradeable_markets, find_best_external_match, expected_value, kelly_fraction

# ---Data Classes---
# A single actionable trading opportunity identified by the engine
@dataclass
class Signal:
    market_id:  str #Polymarket internal market identifier
    question: str # Full text of the prediction market question
    end_date: pd.Timestamp # Market resolution date
    direction: str #"BUY YES" or "BUY NO"
    prob_market: float # Implied probability from Polymarket's current price
    prob_model: float # Calibration model's estimate of P(YES)
    edge: float # prob_model - prob_market (positive = BUY YES, negative = BUY NO)
    kelly_fraction: float # Half-kelly position size as a fraction of bankroll
    spread: float # Current bid-ask spread (our transaction cost)
    liquidity: float # Total liquidity in the market (USD)
    confidence: str #"HIGH" if both sources agree, "MEDIUM" if only one source
    source_model: str = "calibration_rf" # Always present (calibration model)
    token_id_yes: str = None # CLOB token ID, needed for live order execution
    prob_external: float = None # External forecaster consensus P(YES), if found
    source_external: str = None # Platform provided external probability (or None)
    external_title: str = None # Matched question title on the external platform
    external_url: str = None # Link to the matched external question
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc)) # UTC timestamp when this signal was generated

# ---Model training---
# Downloads resolved Polymarket markets and trains the calibration model
# I use resolved markets as labeled data:
# For each closed market I know the actual outcome (YES=1 / NO=0) and the pre-resolution market probability
# The model learns which structural features predict when the market is wrong
def load_and_train_model(max_resolved: int = 3000) -> tuple[Pipeline, list[str]]:
    print("[Model] Fetching resolved markets for training...")
    df_resolved = fetch_resolved_markets(max_markets=max_resolved)

    if df_resolved.empty:
        raise ValueError("No resolved markets returned. Check API connectivity.")
    
    df_feat, feature_cols = build_features(df_resolved)

    # Always use Random Forest, the same model validated in backtester.py
    # train_random_forest() always uses RF with the exact same hyperparameters
    model = train_random_forest(df_feat, feature_cols)

    print(f"[Model] Training complete. Features used: {feature_cols}")
    return model, feature_cols

# ---Signal Generation---
# Applies the trained calibration model to active markets
# Renames columns to match training schema, builds features and runs predict_proba
# Returns a Series of P(YES) estimates aligned with df_tradeable's index
def _get_model_probabilities(model: Pipeline, df_tradeable: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    df = df_tradeable.copy()

    # Active markets use "prob_yes_market"; training used "prob_market"
    if "prob_yes_market" in df.columns and "prob_market" not in df.columns:
        df = df.rename(columns={"prob_yes_market": "prob_market"})

    df_feat, _ = build_features(df)

    valid_mask = df_feat[feature_cols].notna().all(axis=1)
    probs = pd.Series(np.nan, index=df_tradeable.index)

    if valid_mask.sum() > 0:
        X = df_feat.loc[valid_mask, feature_cols]
        probs.loc[valid_mask] = model.predict_proba(X)[:, 1]

    return probs

# Full signal generation pipeline. This is the main function called by bot.py.
def generate_signals(
        model: Pipeline,
        feature_cols: list[str],
        max_markets: int = 300,
        min_edge: float = 0.05,
        max_kelly: float = 0.10,
        use_external: bool =  True,
) -> list[Signal]:
     
    # STEP 1: Fetch active Polymarket markets
    print("\n[Engine] Fetching active markets...")
    df_active = fetch_all_active_markets(max_markets=max_markets)

    # STEP 2: Filter to only tradeable markets (liquidity, spread, price range)
    print("[Engine] Filtering to tradeable markets...")
    df_tradeable = filter_tradeable_markets(df_active)

    if df_tradeable.empty:
        print("[Engine] No tradeable markets found.")
        return []
     
    # STEP 3: Run calibration model to get structural P(YES) estimates
    print("[Engine] Running calibration model...")
    model_probs = _get_model_probabilities(model, df_tradeable, feature_cols)

    signals = []
    total = len(df_tradeable)

    print(f"[Engine] Generating signals for {total} markets"
        f"{' (with external search)' if use_external else ''}")
     
    for i, (idx, row) in enumerate(df_tradeable.iterrows()):
        prob_market = float(row.get("prob_yes_market", row.get("prob_market", np.nan)))
        prob_model = model_probs.loc[idx]
        spread = float(row["spread"]) if pd.notna(row.get("spread")) else 0.02
        liquidity = float(row["liquidity"]) if pd.notna(row.get("liquidity")) else 0.0

        if np.isnan(prob_model) or np.isnan(prob_market):
            continue

        # STEP 4: Optionally search Metaculus / Manifold for external consensus
        prob_external = None
        source_external = None
        external_title = None
        external_url = None
        confidence = "MEDIUM" # Default: only model available

        if use_external:
            print(f"[{i+1}/{total}] Searching: {row["question"][:60]}...", end="\r")
            match = find_best_external_match(row["question"])
            time.sleep(0.35) # Respect Metaculus rate limit (~150 req/hour)

            if match:
                prob_external = match["match_prob"]
                source_external = match["source"]
                external_title = match["match_title"]
                external_url = match["match_url"]

                # HIGH confidence: both sources agree on direction of mispricing
                model_above = prob_model > prob_market
                extern_above = prob_external > prob_market
                if model_above == extern_above:
                    confidence = "HIGH"

        # STEP 5: Compute edge using RF probability alone (consistently with backtest)
        # External sources only upgrade confidence to HIGH, they don't modify the probability or position size
        # Always use prob_model (RF) directly (never blend with external)
        # The backtest only validated the RF. Blending an unvalidated source would make live performance inconsistent with the backtest
        edge = expected_value(prob_model, prob_market)
        direction = "BUY YES" if edge > 0 else "BUY NO"

        # Edge must exceed spread (transaction cost) plus minimum threshold
        if abs(edge) < spread + min_edge:
            continue

        # STEP 6: Apply Kelly criterion for position sizing
        kelly = kelly_fraction(prob_model, prob_market, half_kelly=True)
        kelly = min(kelly, max_kelly) # Cap to prevent oversized positions

        if kelly <= 0: continue
        # STEP 7: Append signal to list
        signals.append(Signal(
            market_id = str(row.get("market_id", "")),
            token_id_yes = str(row.get("token_id_yes", "")),
            question = str(row.get("question", "")),
            end_date = row.get("end_date"),
            direction = direction,
            prob_market = round(prob_market, 4),
            prob_model = round(float(prob_model), 4),
            prob_external = round(prob_external, 4) if prob_external else None,
            edge = round(edge, 4),
            kelly_fraction = round(kelly, 4),
            spread = round(spread, 4),
            liquidity = round(liquidity, 2),
            confidence = confidence,
            source_external = source_external,
            external_title = external_title,
            external_url = external_url,
        ))
    
    # Sort by absolute edge: largest mispricing first
    signals.sort(key=lambda s: abs(s.edge), reverse=True)
    
    print(f"\n[Engine] Done. {len(signals)} signals generated"
          f"({sum(1 for s in signals if s.confidence == "HIGH")} HIGH confidence).")
    return signals

# Converts a list of Signal objects to a pandas DataFrame for logging and analysis
def signals_to_dataframe(signals: list[Signal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    
    rows = []
    for s in signals:
        rows.append({
            "generated_at": s.generated_at.isoformat(),
            "market_id": s.market_id,
            "token_id_yes": s.token_id_yes,
            "question": s.question,
            "end_date": s.end_date,
            "direction": s.direction,
            "prob_market": s.prob_market,
            "prob_model": s.prob_model,
            "prob_external": s.prob_external,
            "edge": s.edge,
            "kelly_fraction": s.kelly_fraction,
            "spread": s.spread,
            "liquidity": s.liquidity,
            "confidence": s.confidence,
            "source_external": s.source_external,
            "external_title": s.external_title,
            "external_url": s.external_url,
        })
    
    return pd.DataFrame(rows)

# ---Main---
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("SIGNAL ENGINE - Standalone Test Run")
    print("=" * 60)

    model, feature_cols = load_and_train_model(max_resolved=2000)

    signals = generate_signals(
        model = model,
        feature_cols = feature_cols,
        max_markets = 200,
        min_edge = 0.05,
        max_kelly = 0.10,
        use_external = False, # Set True to also query Metaculus/Manifold
    )

    if not signals:
        print("\nNo signals found with current filters.")
    else:
        df = signals_to_dataframe(signals)
        print(f"\n{"=" * 60}")
        print(f"TOP SIGNALS:")
        print(f"{"=" * 60}")
        cols = ["question", "direction", "prob_market", "prob_model", "edge", "kelly_fraction", "confidence"]
        print(df[cols].head(10).to_string(index=False))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        path = f"data/signals_{ts}.csv"
        df.to_csv(path, index=False)
        print(f"\nSignals saved to: {path}")
