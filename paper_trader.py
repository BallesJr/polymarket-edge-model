import json
import os
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from signal_engine import Signal

# ---Logging---
logger = logging.getLogger(__name__)

# ---Constants---
PORTFOLIO_PATH = "data/paper_portfolio.json"
TRADES_LOG_PATH = "data/trades_log.csv"

# Position sizing constraints
MAX_KELLY = 0.10 # Never risk more than 10% of bankroll on a single trade
MAX_POSITION_USD = 500.0 # Hard cap per trade regardless of Kelly
LIQUIDITY_TAKE = 0.10 # Max fraction of market liquidity we absorb (slippage control)

# ---Position Dataclass---
# Separating Position from Signal keep concerns clean:
# Signal is read-only market intelligence, Position is a mutable financial record with lifecycle
@dataclass
class Position:
    # Identity
    market_id: str
    token_id_yes: Optional[str]
    question: str
    end_date: str # ISO string for JSON serialisability

    # Trade parameters
    direction: str # "BUY NO" | "BUY YES"
    prob_market: float # Market price at entry
    prob_model: float # RF model estimate at entry
    edge: float # prob_model - prob_market at entry
    kelly: float # Kelly fraction used for sizing
    confidence: str # "HIGH" | "MEDIUM"
    size_usd: float # Capital allocated (USD)
    price_paid: float # Token price paid (prob_market for YES, 1-prob_market for NO)

    # Metadata
    source_model: str = "calibration_rf"
    prob_external: Optional[float] = None
    source_external: Optional[str] = None
    opened_at: str = ""

    # Resolution (filled on close)
    status: str = "OPEN" # "OPEN" | "RESOLVED" | "EXPIRED"
    outcome: Optional[int] = None # 1=YES won, 0=NO won
    pnl: Optional[float] = None
    resolved_at: Optional[str] = None

# ---Portfolio State---

# Load portfolio from disk, or initialise a fresh one on first run
def _load_portfolio(initial_bankroll: float = 10_000.0) -> dict:
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    
    portfolio = {
        "bankroll": initial_bankroll,
        "initial_bankroll": initial_bankroll,
        "open_positions": [],
        "resolved_trades": [],
    }
    _save_portfolio(portfolio)
    return portfolio

def _save_portfolio(portfolio: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)
    
# ---Position Sizing---
# Kelly sizing with three independent caps:
# 1. MAX_KELLY cap - prevents over-concentration under model uncertainty
# 2. Liquidity cap - absorbing >10% of a market's liquidity causes slippage
# 3. MAX_POSITION_USD - absolute hard cap regardless of bankroll size

# Note: confidence does NOT scale size here- we collect that data first, then validate whether HIGH > MEDIUM before acting on it

def _compute_size(bankroll: float, kelly: float, liquidity: Optional[float]) -> float:
    raw = bankroll * min(kelly, MAX_KELLY)
    if liquidity and liquidity > 0:
        raw = min(raw, liquidity * LIQUIDITY_TAKE)
    return round(min(raw, MAX_POSITION_USD), 2)

# ---P&L Calculation---
# Each token pays $1.00 if it wins, $0.00 if it loses
# Tokens held = size_usd / price_paid
# Win: net_gain = (1 - price_paid) / price_paid * size_usd
# Loss: net_loss = -size_usd
def _compute_pnl(pos: Position, outcome: int) -> float:
    won = (pos.direction == "BUY YES" and outcome == 1) or (pos.direction == "BUY NO" and outcome == 0)
    if won:
        return round(pos.size_usd * (1 - pos.price_paid) / pos.price_paid, 2)
    return round(-pos.size_usd, 2)

# ---Trade Entry---
# Evaluate a signal and open a paper position if conditions are met
# Guards:
# - No duplicate positions on the same market_id
# - Computed size must be >= $1 (Kelly near-zero -> no real edge)
# - Sufficient bankroll remaining
def open_position(signal: Signal, portfolio: dict) -> Optional[Position]:
    open_ids = {p["market_id"] for p in portfolio["open_positions"]}
    if signal.market_id in open_ids:
        logger.debug(f"Skipping duplicate: {signal.market_id}")
        return None
    
    # Guard: skip markets whose end_date has already passed
    # Prevents the open -> expire -> reopen loop on stale markets
    end_dt = pd.to_datetime (signal.end_date, utc=True)
    if end_dt < datetime.now(timezone.utc):
        logger.debug(f"Skipping expired market: {signal.question[:50]}")
        return None
    
    size = _compute_size(portfolio["bankroll"], signal.kelly_fraction, signal.liquidity)
    if size < 1.0:
        logger.debug(f"Size too small (${size:.2f}): {signal.question[:50]}")
        return None
    if size > portfolio["bankroll"]:
        size = round(portfolio["bankroll"], 2)
    if size < 1.0:
        return None
    
    price_paid = signal.prob_market if signal.direction == "BUY YES" else (1 - signal.prob_market)

    pos = Position(
        market_id = signal.market_id,
        token_id_yes = signal.token_id_yes,
        question =  signal.question,
        end_date = str(signal.end_date),
        direction =  signal.direction,
        prob_market =  signal.prob_market,
        prob_model = signal.prob_model,
        edge =  signal.edge,
        kelly = signal.kelly_fraction,
        confidence =  signal.confidence,
        size_usd = size,
        price_paid = round(price_paid, 4),
        source_model = signal.source_model,
        prob_external = signal.prob_external,
        source_external = signal.source_external,
        opened_at = datetime.now(timezone.utc).isoformat(),
    )

    portfolio["bankroll"] = round(portfolio["bankroll"] - size, 2)
    portfolio["open_positions"].append(asdict(pos))
    _save_portfolio(portfolio)

    logger.info(
        f"OPENED {pos.direction:8s} | ${pos.size_usd:.0f} | "
        f"edge={pos.edge:+.3f} | conf={pos.confidence} | "
        f"{pos.question[:55]}"
    )
    return pos

# ---Trade Resolution---
# Resolve an open position once the market outcome is known
def resolve_position(market_id: str, outcome: int, portfolio: dict) -> Optional[dict]:
    for i, pos_dict  in enumerate(portfolio["open_positions"]):
        if pos_dict["market_id"] != market_id:
            continue

        pos = Position(**pos_dict)
        pnl = _compute_pnl(pos, outcome)

        pos_dict.update({
            "status": "RESOLVED",
            "outcome": outcome,
            "pnl": pnl,
            "resolved_at": datetime.now(timezone.utc).isoformat()
        })

        portfolio["bankroll"] = round(portfolio["bankroll"] + pos.size_usd + pnl, 2)
        portfolio["open_positions"].pop(i)
        portfolio["resolved_trades"].append(pos_dict)

        _save_portfolio(portfolio)
        _append_trade_log(pos_dict)

        result = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            f"RESOLVED [{result}] P&L={pnl:+.2f} | "
            f"{pos_dict["direction"]:8s} | {pos_dict["question"][:55]}"
        )
        return pos_dict
    
    logger.warning(f"resolve_position: market_id {market_id} not found in open positions.")
    return None

# Close positions whose end_date has passed with no resolution
def expire_stale_positions(portfolio: dict) -> list[dict]:
    now = datetime.now(timezone.utc)
    expired, remaining = [], []

    for pos_dict  in portfolio["open_positions"]:
        try:
            end_dt = pd.to_datetime(pos_dict["end_date"], utc=True)
        except Exception:
            remaining.append(pos_dict)
            continue

        if end_dt < now:
            pos_dict.update({
                "status": "EXPIRED",
                "pnl": 0.0,
                "resolved_at": now.isoformat(),
            })
            portfolio["bankroll"] = round(portfolio["bankroll"] + pos_dict["size_usd"], 2)
            portfolio["resolved_trades"].append(pos_dict)
            expired.append(pos_dict)
            logger.info(f"EXPIRED (capital returned): {pos_dict["question"][:60]}")
        else:
            remaining.append(pos_dict)
    
    portfolio["open_positions"] = remaining
    if expired:
        _save_portfolio(portfolio)

    return expired

# ---Trade Log---
def _append_trade_log(trade: dict) -> None:
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([trade])
    header = not os.path.exists(TRADES_LOG_PATH)
    df.to_csv(TRADES_LOG_PATH, mode="a", header=header, index=False)

# ---Portfolio Metrics---
def compute_metrics(portfolio: dict) -> dict:
    initial = portfolio["initial_bankroll"]
    bankroll = portfolio["bankroll"]
    open_pos = portfolio.get("open_positions", [])
    resolved = portfolio.get("resolved_trades", [])
    open_exposure = sum(p["size_usd"] for p in open_pos)
    total_equity = bankroll + open_exposure

    base = {
        "total_trades": len(resolved),
        "open_positions": len(open_pos),
        "open_exposure_usd": round(open_exposure, 2),
        "bankroll": round(bankroll, 2),
        "total_equity": round(total_equity, 2),
        "roi_pct": round((total_equity - initial) / initial * 100, 2),
    }

    if not resolved:
        return {**base, "win_rate_pct": None, "profit_factor": None, "max_drawdown_pct": None, "avg_edge": None}
    
    df = pd.DataFrame(resolved)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    wins = (df["pnl"] > 0).sum()
    gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
    equity_curve = initial + df["pnl"].cumsum().values
    running_max = np.maximum.accumulate(equity_curve)
    max_dd = ((equity_curve - running_max) / running_max).min() * 100

    return {
        **base,
        "win_rate_pct": round(wins / len(df) * 100, 1),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_edge": round(df["edge"].abs().mean(), 4) if "edge" in df.columns else None,
    }

def print_summary(portfolio: dict) -> None:
    m = compute_metrics(portfolio)
    print("\n" + "="*55)
    print("PAPER PORTFOLIO")
    print("="*55)
    print(f"Bankroll (free): ${m["bankroll"]:>10,.2f}")
    print(f"Open exposure: ${m["open_exposure_usd"]:>10,.2f}")
    print(f"Total equity: ${m["total_equity"]:>10,.2f}")
    print(f"ROI: {m["roi_pct"]:>+10.2f}%")
    print("-"*55)
    print(f"Resolved trades: {m["total_trades"]}")
    print(f"Open positions: {m["open_positions"]}")
    if m["win_rate_pct"] is not None:
        print(f"Win rate: {m["win_rate_pct"]:>10.1f}%")
        print(f"Profit factor: {m["profit_factor"]:>10.2f}")
        print(f"Max drawdown: {m["max_drawdown_pct"]:>+10.2f}%")
    print("="*55)
    if portfolio["open_positions"]:
        print("\n OPEN POSITIONS:")
        for p in portfolio["open_positions"]:
            print(f"[{p["direction"]:8s}] ${p["size_usd"]:>5.0f} | "
                  f"edge={p["edge"]:+.3f} | conf={p["confidence"]} | "
                  f"{p["question"][:50]}")

# ---Public API---
# Entry point called by bot.py. 
# Opens positions for new signals, expires stale positions, and prints a portfolio summary.
def process_signals(signals: list[Signal], initial_bankroll: float = 10_000.0,) -> dict:
    portfolio =  _load_portfolio(initial_bankroll)

    expired = expire_stale_positions(portfolio)
    if expired:
        logger.info(f"{len(expired)} position(s) expired (capital returned).")

    opened = skipped = 0
    for signal in signals:
        pos = open_position(signal, portfolio)
        if pos:
            opened += 1
        else:
            skipped += 1
    
    logger.info(f"Cycle complete - opened: {opened} | skipped:{skipped}")
    print_summary(portfolio)
    return portfolio

# ---Quick Test---
if __name__ == "__main__":
    fake = Signal(
        market_id = "test-001",
        question = "Will the Fed cut rates in June 2026",
        end_date = pd.Timestamp("2026-06-01", tz="UTC"),
        direction = "BUY YES",
        prob_market = 0.38,
        prob_model = 0.54,
        edge = 0.16,
        kelly_fraction = 0.09,
        spread = 0.02,
        liquidity = 8000.0,
        confidence = "HIGH",
        token_id_yes = "fake-token-001",
    )

    portfolio = process_signals([fake], initial_bankroll=10_000.0)

    # Simulate a WIN resolution
    resolve_position("test-001", outcome=1, portfolio=portfolio)
    print_summary(portfolio)