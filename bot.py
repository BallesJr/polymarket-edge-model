import argparse
import logging
import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone

from signal_engine import load_and_train_model, generate_signals
from paper_trader import process_signals, resolve_position, expire_stale_positions, print_summary, compute_metrics, _load_portfolio

# ---Configuration---
RUN_INTERVAL_MINUTES = 30 # How often the full pipeline run
MAX_MARKETS = 300 # Active markets scanned per cycle
INITIAL_BANKROLL = 10_000.0
USE_EXTERNAL = False # Set True to also query Metaculus/Manifold

# ---Logging---
# Two handlers: console (INFO) and file (DEBUG)
# The file keeps a full audit trail; the console shows only whats matters
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/bot.log"), logging.StreamHandler(),]
)

# Suppress verbose third-party logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---Gamma API---
GAMMA_URL = "https://gamma-api.polymarket.com"

# ---Resolution Checker---
# Called every cycle for each open position. Queries the Gamma API to check whether the market has resolved and with what outcome
# The Gamma API returns outcome prices = ["1.0", "0.0"] for YES wins, ["0.0", "1.0"] for NO wins, and ["0.5", "0.5"] for cancelled markets

# Query the Gamma API for the resolution outcome of a market
def _fetch_outcome(market_id: str) -> int | None:
    try:
        resp = requests.get(f"{GAMMA_URL}/markets/{market_id}", timeout=8)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        if data.get("active", True):
            return None # Market still open
        
        import json as _json
        prices = _json.loads(data.get("outcomePrices", "[]"))
        if len(prices) < 2:
            return None
        
        p0, p1 = float(prices[0]), float(prices[1])
        if p0 == 1.0 and p1 == 0.0:
            return 1 # YES won
        if p0 == 0.0 and p1 == 1.0:
            return 0 # NO won
        return None # Cancelled or ambiguous
    
    except (requests.RequestException, ValueError):
        return None

# Check all open positions against the Gamma API and resolve any that have reached a conclusion
def check_resolutions(portfolio: dict) -> list[dict]:
    open_positions = portfolio.get("open_positions", [])
    if not open_positions:
        return []
    
    logger.info(f"Checking {len(open_positions)} open position(s) for resolution...")
    resolved_this_cycle = []

    # Iterate over a copy
    # I modify the list inside resolve_position
    for pos in open_positions[:]:
        outcome = _fetch_outcome(pos["market_id"])
        if outcome is not None:
            trade = resolve_position(pos["market_id"], outcome, portfolio)
            if trade:
                resolved_this_cycle.append(trade)
        time.sleep(0.3)

    return resolved_this_cycle

# ---Metrics Snapshot---
# Appended to a JSONL file after every cycle (one line per run)
# Lets you reconstruct the full equity curve and track performance over time even across bot restarts
def _save_metrics_snapshot(portfolio: dict) -> None:
    metrics = compute_metrics(portfolio)
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open("data/metrics_history.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")

# ---Single Cycle---
# Extracted into its own function so it can be called both in --loop mode and in single-run mode without duplicating logic

# One full bot cycle
def run_cycle(model, feature_cols: list[str]) -> dict:
    cycle_start = datetime.now(timezone.utc)
    logger.info(f"---Cycle start: {cycle_start.strftime('%Y-%m-%d %H:%M UTC')}---")

    # STEP 1: Generate signals
    try:
        signals = generate_signals(
            model = model,
            feature_cols = feature_cols,
            max_markets = MAX_MARKETS,
            min_edge = 0.05,
            max_kelly = 0.10,
            use_external = USE_EXTERNAL,
        )
        logger.info(f"Signals generated: {len(signals)}")
    except Exception as e:
        logger.error(f"Signal engine failed: {e}", exc_info=True)

        signals = []

    # STEP 2: Check resolutions before opening new positions, frees up capital from closed positions to use for new ones
    portfolio = _load_portfolio(INITIAL_BANKROLL)
    resolved = check_resolutions(portfolio)
    if resolved:
        logger.info(f"Resolved {len(resolved)} position(s) this cycle.")

    # STEP 3: Open new positions
    portfolio = process_signals(signals, initial_bankroll=INITIAL_BANKROLL)

    # STEP 4: Persist metrics
    _save_metrics_snapshot(portfolio)

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    logger.info(f"---Cycle complete in {elapsed:.1f}s---\n")

    return portfolio

# ---CLI (Command Line Interface)---
def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Paper Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python bot.py Run one cycle and exit
        python bot.py --loop Run every 30 minutes continuously
        python bot.py --status Print current portfolio and exit
        python bot.py --resolve-only Check open position resolutions and exit
        """
    )
    parser.add_argument("--loop",
                        action="store_true",
                        help="Run continuously every RUN_INTERVAL_MINUTES")
    parser.add_argument("--status",
                        action="store_true",
                        help="Print portfolio status and exit")
    parser.add_argument("--resolve-only",
                        action="store_true",
                        help="Check resolutions only, no new signals")
    args = parser.parse_args()

    # --status: just print and exit
    if args.status:
        portfolio = _load_portfolio(INITIAL_BANKROLL)
        print_summary(portfolio)
        return
    
    # --resolve-only: check resolutions and exit
    if args.resolve_only:
        portfolio = _load_portfolio(INITIAL_BANKROLL)
        resolved = check_resolutions(portfolio)
        expired = expire_stale_positions(portfolio)
        logger.info(f"Resolved: {len(resolved)} | Expired: {len(expired)}")
        print_summary(portfolio)
        return
    
    # All other modes require the module - train once, reuse across cycles
    logger.info("Loading / training calibration model...")
    model, feature_cols = load_and_train_model(max_resolved=3000)

    # Single run
    if not args.loop:
        run_cycle(model, feature_cols)
        return
    
    # Continuous loop
    logger.info(f"Starting loop - cycle every {RUN_INTERVAL_MINUTES} min. Ctrl+C to stop.\n")
    while True:
        try:
            run_cycle(model, feature_cols)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unhandled error in cycle: {e}", exc_info=True) # Log but keep running, transient API errors shouldn't kill the bot

            logger.info(f"Sleeping {RUN_INTERVAL_MINUTES} min...")
            time.sleep(RUN_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main()
