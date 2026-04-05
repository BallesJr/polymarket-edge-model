import pandas as pd
import numpy as np
import requests 
import time
from datetime import datetime, timezone

from polymarket_api import fetch_all_active_markets

# Filters the markets with extreme price (>0.05 or <0.95), with too spread, with no liquidity...
def filter_tradeable_markets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Some columns may come as strings - force numeric to enable comparisons
    for col in ["prob_yes_market", "spread", "liquidity", "volume_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (
        (df["prob_yes_market"].between(0.05, 0.95)) & # Skip extreme probabilities - hard to find edge
        (df["spread"].notna()) & (df["spread"] <= 0.05) &   # Skip if there is too spread (> 5%)
        (df["liquidity"].notna()) & (df["liquidity"] >= 500) & # Force more than 500$ in liquidity
        (df["neg_risk"] == False) & # Skip if market is special with different mechanics
        (df["accepting_orders"] == True) # Only markets where it's possible place orders
    )

    result = df[mask].copy()
    print(f"Total market {len(df)} -> Operables: {len(result)}")

    return result

# Searches probabilities from external sources
# Searches Metaculus for similar question to the Polymarket one and returns probability and metadata
def search_metaculus(question: str, max_results: int = 3) -> list[dict]:
    url = "https://www.metaculus.com/api2/questions/"
    params = {
        "search": question[:100], # Truncate to avoid overly long URLs
        "status": "open",
        "type": "forecast",
        "order_by": "-votes", # Most voted first
        "limit": max_results,
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
        matches = []
        for q in results:
            # community_prediction.full.q2 is the median of all community forecasts
            pred = q.get("community_prediction", {})
            prob = pred.get("full", {}).get("q2") if pred else None
            if prob is None:
                continue
            matches.append({
                "source": "metaculus",
                "match_title": q.get("title"),
                "match_prob": round(float(prob), 4),
                "match_url": f"https://www.metaculus.com/questions/{q.get('id')}",
                "match_votes": q.get("number_of_forecasters", 0),
            })

        return matches
    except requests.RequestException:
        return []

# Searches Manifold Markets for binary markets similar to the Polymarket question    
def search_manifold(question: str, max_results: int = 5) -> list[dict]:
    url = "https://api.manifold.markets/v0/search-markets"
    params = {
        "term": question[:100],
        "filter": "open",
        "sort": "liquidity", # Most liquidity
        "limit": max_results,
        "contractType": "BINARY", # Only YES/NO markets, same structure as we take from Polymarket
    }
    
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        results = resp.json()
        matches = []

        for  m in results:
            prob = m.get("probability") # On Manifold, market price is the probability
            if prob is None:
                continue

            bettors = m.get("uniqueBettorCount",0)
            
            matches.append({
                "source": "manifold",
                "match_title": m.get("question"),
                "match_prob": round(float(prob), 4),
                "match_url": m.get("url", ""),
                "match_votes": bettors, # Number of unique bettors
            })
            
        return matches
    except requests.RequestException:
        return []

# Returns the best coincidence with the Polymarket question (compare Metaculus and Manifold, returns the match with the most participants)
def find_best_external_match(question: str) -> dict | None:
    matches = search_metaculus(question, max_results=5)
    if not matches:
        matches = search_manifold(question, max_results=5) # Fallback to Manifold
    if not matches:
        return None
    
    best = max(matches, key=lambda x: x["match_votes"]) # Most participants
    if best["match_votes"] < 5:
        return None # Too few forecasters
    return best      

# Calculates EV (Expected Value) = prob_model - prob_market, if it's positive means buy YES. negative means buy NO
def expected_value(prob_model: float, prob_market: float) -> float:
    return round(prob_model - prob_market,  4)

# Uses Kelly's criterion to decide how much % you should risk (half-Kelly by default)
def kelly_fraction(prob_model: float, prob_market: float, half_kelly: bool = True) -> float:
    if prob_market <= 0 or prob_market >= 1:
        return 0.0
    
    p = prob_model # Estimated probability of winning
    q = 1 - p   # Estimated probability of losing

    if p >= 0.5:
        # Buy YES: b is net gain per unit if YES wins
        b = (1 - prob_market) / prob_market
        f = (p * b - q) / b
    else:
        # Buy NO: b is calculated from the NO side
        price_no = 1 - prob_market 
        b = prob_market / price_no # Net gain per unit if NO wins
        f = (q * b - p) / b # Kelly formula from NO perspective

    if f <= 0:
        return 0.0 # No mathematical edge
    
    # Half-Kelly as a safety margin
    return round((f/2 if half_kelly else f), 4)

# Full pipeline: filters markets -> Searches from external sources -> Calculates edge and Kelly sizing
def run_pipeline(df: pd.DataFrame, min_edge: float = 0.05) -> pd.DataFrame:
    #Step 1: Filters tradeable markets
    df_tradeable = filter_tradeable_markets(df)
    if df_tradeable.empty:
        print(f"No tradeable markets found.")
        return pd.DataFrame()
    
    signals = []
    total = len(df_tradeable)

    print(f"Searching external matches for {total} markets...")
    print(f"(May take 1-2 minutes to respect rate limits)\n")

    for i, (_, row) in enumerate(df_tradeable.iterrows()):
        print(f" [{i+1}/{total}] {row["question"][:65]}...", end="\r")
        match = find_best_external_match(row["question"])
        if match is None:
            time.sleep(0.3)
            continue # No reliable external probability found

        prob_model = match["match_prob"]
        prob_market = row["prob_yes_market"]
        spread = row["spread"] if pd.notna(row["spread"]) else 0.02

        ev = expected_value(prob_model, prob_market)
        kelly = kelly_fraction(prob_model, prob_market)
        direction = "BUY YES" if ev > 0 else "BUY NO"

        # Edge must exceed spread + minimum threshold to be worth trading
        if abs(ev) < spread + min_edge:
            time.sleep(0.3)
            continue
        signals.append({
            "market_id": row["market_id"],
            "question": row["question"],
            "end_date": row["end_date"],
            "prob_market": prob_market,
            "prob_model": prob_model,
            "edge": ev,
            "kelly_fraction": kelly,
            "direction": direction,
            "spread": spread,
            "liquidity": row["liquidity"],
            "volume_total": row["volume_total"],
            "competitive": row["competitive"],
            "source": match["source"], # Which platform provided the external probability
            "match_title": match["match_title"], # Exact title of the matched question
            "match_votes": match["match_votes"],
            "match_url": match["match_url"],
        })

        time.sleep(0.4) # Metaculus rate limit: ~150 req/hour

    if not signals:
        print("\nNo opportunities found with the current filter.")
        return pd.DataFrame()

    df_signals = pd.DataFrame(signals) \
                .sort_values("edge", key=abs, ascending=False) \
                .reset_index(drop=True)
    return df_signals

# Main
if __name__ == "__main__":
    print("=== Polymarket edge model ===\n")

    print("Loading markets...")
    df_markets = fetch_all_active_markets(max_markets=300)

    df_signals = run_pipeline(df_markets, min_edge=0.05)
    if df_signals.empty:
        print("No signals.")
    else:
        print(f"\n{"="*60}\n")
        print(f"SIGNALS FOUND: {len(df_signals)}")
        print(f"\n{"="*60}\n")

        cols_print = ["question", "prob_market", "prob_model", "edge", "kelly_fraction", "direction", "spread", "source", "match_title", "match_url"]
        print(df_signals[cols_print].to_string(index=False))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        path = f"data/signals_{ts}.csv"
        df_signals.to_csv(path, index=False)
        print(f"\nSignals saved in: {path}")       