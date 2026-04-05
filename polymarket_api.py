import requests
import pandas as pd
import json
import time
from datetime import datetime, timezone

# URL API bases
GAMMA_URL = "https://gamma-api.polymarket.com" # Metadata: titles, rules, categories...
CLOB_URL  = "https://clob.polymarket.com" # Markets: real-time orderbook, midpoint...

# Makes a single paginated call to the GAMMA API to get a markets list as raw JSON
def fetch_markets(limit: int = 50, offset: int = 0, active_only: bool = True) -> list[dict]:
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true" if active_only else "false", 
        "closed": "false" 
    }

    resp = requests.get(f"{GAMMA_URL}/markets", params=params, timeout=10)
    resp.raise_for_status() # Raises exception on HTTP error (4xx, 5xx)
    return resp.json()

# Paginates fetch_markets until reaching max_markets
def fetch_all_active_markets(max_markets: int = 500) -> pd.DataFrame:
    all_markets = []
    offset = 0
    batch = 100 # Max recommended per call by the GAMMA API

    print(f"Loading markets...")

    while len(all_markets) < max_markets:
        batch_data = fetch_markets(limit=batch, offset=offset)
        if not batch_data:
            break # API returned no more markets

        all_markets.extend(batch_data)
        offset += len(batch_data) # Advance cursor for next page

        print(f" -> {len(all_markets)} loaded markets", end="\r")

        if len(batch_data) < batch:
            break # Last page: fewer results than requested

        time.sleep(0.3) # Soft rate limit to avoid API blocks
    print(f"\nTotal: {len(all_markets)} markets found.")
    return _markets_to_dataframe(all_markets)

# Converts raw JSON markets list into a clean dataframe
def _markets_to_dataframe(markets: list[dict]) -> pd.DataFrame:
    rows = []
    for m in markets:
        try:
            prices = json.loads(m.get("outcomePrices", "[]")) #outcomePrices come as a JSON string, needs parsing
            prob_yes = float(prices[0]) if prices else None
            prob_no = float(prices[1]) if len(prices) > 1 else None
        except (json.JSONDecodeError, ValueError, IndexError):
            prob_yes = prob_no = None

        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]")) # clobTokenIds comes as a JSON too
            token_yes = token_ids[0] if token_ids else None
        except (json.JSONDecodeError, ValueError, IndexError):
            token_yes = None
        
        # category lives inside "tags"
        tags = m.get("tags") or []
        if isinstance(tags, list) and tags:
            category = tags[0].get("label") if isinstance(tags[0], dict) else str(tags[0])
        else:
            category = m.get("category") # Fallback in case API structure changes
        
        rows.append({
            "market_id": m.get("id"),
            "question": m.get("question"),
            "end_date": m.get("endDate"),
            "prob_yes_market": prob_yes, # Implied YES probability (0-1)
            "prob_no_market": prob_no,
            "best_bid": m.get("bestBid"),
            "best_ask": m.get("bestAsk"),
            "spread": m.get("spread"),
            "last_trade_price": m.get("lastTradePrice"),
            "change_1d": m.get("oneDayPriceChange"),
            "change_1w": m.get("oneWeekPriceChange"),
            "volume_24h": m.get("volume24hr"),
            "volume_total": m.get("volume"),
            "liquidity": m.get("liquidity"),
            "competitive": m.get("competitive"), # 0-1 for market activity level
            "token_id_yes": token_yes,  # Needed to query the CLOB directly
            "accepting_orders": m.get("acceptingOrders", False),
            "neg_risk": m.get("negRisk", False), # Needs to avoid special markets where YES+NO != 1
        })

    df = pd.DataFrame(rows)

    # Converts to datetime so we can calculate days until expiry
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    return df

# Returns the midpoint price for a token from the clob (more accurate than outcomePrices)
def fetch_midpoint(token_id: str) -> float | None:
    resp = requests.get(f"{CLOB_URL}/midpoint", params={"token_id": token_id}, timeout=5)
    if resp.status_code != 200:
        return None
    data = resp.json()
    try:
        return float(data.get("mid", None))
    except (TypeError, ValueError):
        return None

# Create a orderbook's summary (returns the best bids, the best ask and the spread for a token)
def fetch_orderbook_summary(token_id: str) -> dict:
    resp = requests.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=5)
    if resp.status_code != 200:
        return {}
    
    book = resp.json()
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    # max() for bids and min() for asks because the API does not guarantee order
    best_bid = max((float(b["price"]) for b in bids), default=None)
    best_ask = min((float(a["price"]) for a in asks), default=None)

    spread = round(best_ask - best_bid, 4) if (best_bid is not None and best_ask is not None) else None

    return{
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread
    }

# Enriches adding the midpoint and the spread from the CLOB to the N markets with the most liquidity
def enrich_with_clob_data(df: pd.DataFrame, max_markets: int =  50) -> pd.DataFrame:
    df_sorted = df.dropna(subset=["token_id_yes", "volume_total"]) \
                .sort_values("volume_total", ascending=False) \
                .head(max_markets) \
                .copy()
    
    midpoints = []
    spreads = []

    print(f"Consulting CLOB for the {len(df_sorted)} most liquid markets...")
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        mid = fetch_midpoint(row["token_id_yes"])
        book = fetch_orderbook_summary(row["token_id_yes"])

        midpoints.append(mid)
        spreads.append(book.get("spread"))

        print(f" -> {i+1}/{len(df_sorted)}", end="\r")
        time.sleep(0.15) # ~6 req/s max recommended

    df_sorted["midpoint_clob"] = midpoints
    df_sorted["spread_clob"] = spreads

    print(f"\nDone!")
    return df_sorted

# Saves a timestamped CSV snapshot
def save_snapshot(df: pd.DataFrame, path: str = "data/markets_snapshot.csv") -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df["snapshot_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(path, index=False)

    print(f"Snapshot saved to: {path}")

# Test
if __name__ == "__main__":
    # Step 1: Downloads active markets
    df_markets = fetch_all_active_markets(max_markets=200)

    print(f"\n{"="*55}")
    print(f"Active markets found: {len(df_markets)}")
    print(f"Columns: {list(df_markets.columns)}")

    print(f"\nTop 5 volume markets:")
    top5 = df_markets.dropna(subset=["volume_total"]) \
            .sort_values("volume_total", ascending=False) \
            .head(5)[["question", "prob_yes_market", "spread", "volume_total", "liquidity"]]
    print(top5.to_string(index=False))
    
    print(f"\nTop 10 markets withs spread + price change:")
    cols_show = ["question", "prob_yes_market", "best_bid", "best_ask", "spread", "change_1d", "change_1w"]
    df_liquid = df_markets.dropna(subset=["spread"]) \
                .sort_values("volume_total", ascending=False) \
                .head(10)
    print(df_liquid[cols_show].to_string(index=False))
    # Step 2: Save snapshot for future backtesting
    save_snapshot(df_markets, path="data/markets_snapshot.csv")