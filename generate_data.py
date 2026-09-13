"""
generate_data.py
-----------------
Generates a realistic multi-stock historical OHLCV dataset for the
Stock Market Trend Analysis project.

NOTE ON DATA SOURCE:
This project is designed to work with REAL data pulled from Yahoo Finance
via the `yfinance` library. To fetch real data on your own machine, run:

    import yfinance as yf
    df = yf.download(["AAPL", "MSFT", "AMZN", "TSLA"], start="2022-01-01", end="2024-12-31")

Since this sandbox environment has no internet access to financial data
APIs, this script generates a statistically realistic synthetic dataset
(using Geometric Brownian Motion with drift/volatility calibrated to look
like real equities) in the EXACT same schema real Yahoo Finance data has:
Date, Ticker, Open, High, Low, Close, Adj Close, Volume.

All analysis code in the notebook works unchanged on either dataset —
just swap out this file's output for a real yfinance pull.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ----------------------------------------------------------------------
# Configuration: 5 stocks across different sectors, each with a distinct
# "personality" (trend + volatility) so the analysis has real contrasts
# ----------------------------------------------------------------------
STOCKS = {
    "TECHCO":   {"start_price": 150, "annual_drift": 0.22,  "annual_vol": 0.32, "sector": "Technology"},
    "RETAILX":  {"start_price": 80,  "annual_drift": 0.08,  "annual_vol": 0.28, "sector": "Retail"},
    "ENERGYCO": {"start_price": 60,  "annual_drift": 0.05,  "annual_vol": 0.35, "sector": "Energy"},
    "BANKFIN":  {"start_price": 45,  "annual_drift": 0.10,  "annual_vol": 0.22, "sector": "Finance"},
    "PHARMAB":  {"start_price": 110, "annual_drift": 0.14,  "annual_vol": 0.20, "sector": "Healthcare"},
}

START_DATE = "2022-01-03"
END_DATE = "2024-12-31"
TRADING_DAYS_PER_YEAR = 252


def generate_gbm_prices(start_price, annual_drift, annual_vol, n_days):
    """Generate a price path using Geometric Brownian Motion, with a
    couple of realistic 'market shock' events injected."""
    dt = 1 / TRADING_DAYS_PER_YEAR
    mu = annual_drift
    sigma = annual_vol

    shocks = np.random.normal(
        loc=(mu - 0.5 * sigma ** 2) * dt,
        scale=sigma * np.sqrt(dt),
        size=n_days,
    )

    # Inject 2 realistic shock events (e.g. earnings surprise / market correction)
    shock_day_1 = int(n_days * 0.35)
    shock_day_2 = int(n_days * 0.72)
    shocks[shock_day_1] += np.random.choice([-1, 1]) * np.random.uniform(0.06, 0.10)
    shocks[shock_day_2] += np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.09)

    log_returns = shocks
    price_path = start_price * np.exp(np.cumsum(log_returns))
    return np.insert(price_path, 0, start_price)[:-1]


def build_ohlcv(close_prices, base_volume):
    """Derive Open/High/Low/Volume around a close price series, the way
    real daily bars relate to each other."""
    n = len(close_prices)
    daily_noise = np.random.uniform(0.003, 0.018, n)

    open_prices = close_prices * (1 + np.random.normal(0, 0.004, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + daily_noise)
    low_prices = np.minimum(open_prices, close_prices) * (1 - daily_noise)

    # Volume tends to spike on big price-move days
    pct_change = np.abs(np.diff(close_prices, prepend=close_prices[0]) / close_prices)
    volume = base_volume * (1 + pct_change * 15) * np.random.uniform(0.7, 1.3, n)

    return open_prices, high_prices, low_prices, volume.astype(int)


def main():
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)  # business days only
    n_days = len(dates)

    all_rows = []
    for ticker, params in STOCKS.items():
        close = generate_gbm_prices(
            params["start_price"], params["annual_drift"], params["annual_vol"], n_days
        )
        base_volume = np.random.randint(2_000_000, 9_000_000)
        open_, high, low, volume = build_ohlcv(close, base_volume)

        df = pd.DataFrame({
            "Date": dates,
            "Ticker": ticker,
            "Sector": params["sector"],
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Adj Close": np.round(close, 2),
            "Volume": volume,
        })
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    # Inject a small amount of realistic messiness for the cleaning step
    # (a handful of missing values + one duplicate row) — mirrors real raw exports
    missing_idx = np.random.choice(full_df.index, size=15, replace=False)
    full_df.loc[missing_idx, "Volume"] = np.nan
    full_df = pd.concat([full_df, full_df.sample(3, random_state=1)], ignore_index=True)

    full_df = full_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    full_df.to_csv("data/stock_prices_raw.csv", index=False)
    print(f"Generated {len(full_df):,} rows across {len(STOCKS)} tickers")
    print(full_df.head())


if __name__ == "__main__":
    main()
