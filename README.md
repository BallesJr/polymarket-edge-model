# POLYMARKET EDGE MODEL

Polymarket is a decentralized prediction market where people trade on the outcomes of real-world events. Each market has an implied probability — but that probability is not always right.

This project builds a systematic pipeline to find mispriced markets using two complementary approaches: comparing Polymarket prices against external forecasting platforms, and training a machine learning model on thousands of historical resolved markets.

---

## WHAT I WORKED ON

- **API integration**: Connected to Polymarket's public Gamma and CLOB APIs to fetch live markets, order book data, spreads, and price history (no authentication required).
- **External probability comparison**: Searched Metaculus and Manifold Markets for questions matching each Polymarket market, then calculated the edge between platforms using the Kelly Criterion for position sizing.
- **Calibration model**: Downloaded 3,000 resolved markets (with known YES/NO outcomes) and trained a Random Forest to detect whether Polymarket systematically mis-prices certain types of markets.
- **Brier Score analysis**: Evaluated the model using proper probabilistic scoring and generated a Reliability Diagram showing where Polymarket's calibration breaks down.

## PROJECT STRUCTURE

- `polymarket_api.py`: Polymarket API client, it fetches active markets and order book data.
- `expected_value.py`: External comparison pipeline (Metaculus + Manifold + Kelly sizing).
- `calibration.py`: ML calibration model (training, evaluation, and signal generation).

## RESULTS

The calibration model was trained on **3,000 resolved markets** from 2024–2025:

| Model               | Brier Score |
| ------------------- | ----------- |
| Naive (always 50%)  | 0.2500      |
| Polymarket raw      | 0.2482      |
| Logistic Regression | 0.2312      |
| **Random Forest**   | **0.2161**  |

The Random Forest achieves a **12.9% improvement** over the raw market probability, suggesting Polymarket has systematic calibration biases; particularly the longshot bias (overpricing unlikely events).

![Reliability Diagram](data/calibration_analysis.png)

## LIMITATIONS

**Comparison with external resources**: External matches in `expected_value.py` rely on text search and can return semantically similar but non-identical questions. Always verify the `match_title` before acting on a signal.

**Backtesting**: Backtesting is not yet implemented - the model has not been validated with real P&L tracking.

**Bias**: The calibration model shows a BUY YES bias when applied to active markets, likely due to the 3:1 NO/YES class imbalance in the training data.

**Automation**: Order execution is manual. A future `bot/executor.py` module will automate trades via Polymarket's authenticated CLOB API.

## REQUIREMENTS

`pip install requests pandas numpy scikit-learn matplotlib`

## EXECUTION

```bash
python polymarket_api.py    # download active markets
python expected_value.py    # external probability signals
python calibration.py       # ML calibration signals
```
