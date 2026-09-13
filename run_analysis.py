"""
run_analysis.py
Full stock market trend analysis pipeline:
1. Load & clean raw data
2. Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. Compute returns, volatility, Sharpe ratio, max drawdown
4. Correlation analysis across stocks
5. Trend classification
6. Save all charts to /charts and a clean dataset to /data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

RISK_FREE_RATE = 0.05  # annual, for Sharpe ratio


# ------------------------------------------------------------------
# 1. LOAD & CLEAN
# ------------------------------------------------------------------
def load_and_clean(path="data/stock_prices_raw.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])

    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes_removed = n_before - len(df)

    missing_before = df["Volume"].isna().sum()
    # Forward-fill missing volume within each ticker (realistic approach
    # for a low count of missing values in a time series)
    df = df.sort_values(["Ticker", "Date"])
    df["Volume"] = df.groupby("Ticker")["Volume"].transform(lambda s: s.ffill().bfill())
    df["Volume"] = df["Volume"].astype(int)

    print("=== DATA CLEANING SUMMARY ===")
    print(f"Duplicate rows removed : {n_dupes_removed}")
    print(f"Missing Volume filled  : {missing_before}")
    print(f"Final row count        : {len(df):,}")
    print(f"Date range             : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Tickers                : {sorted(df['Ticker'].unique())}\n")

    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 2. TECHNICAL INDICATORS
# ------------------------------------------------------------------
def add_indicators(df):
    df = df.sort_values(["Ticker", "Date"]).copy()
    out = []
    for ticker, g in df.groupby("Ticker"):
        g = g.copy()
        g["SMA20"] = g["Close"].rolling(20).mean()
        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["EMA20"] = g["Close"].ewm(span=20, adjust=False).mean()

        # Daily returns
        g["Daily_Return"] = g["Close"].pct_change()

        # Rolling volatility (annualized, 20-day window)
        g["Volatility_20D"] = g["Daily_Return"].rolling(20).std() * np.sqrt(252)

        # RSI (14-day)
        delta = g["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        g["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = g["Close"].ewm(span=12, adjust=False).mean()
        ema26 = g["Close"].ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        g["BB_Mid"] = g["SMA20"]
        g["BB_Std"] = g["Close"].rolling(20).std()
        g["BB_Upper"] = g["BB_Mid"] + 2 * g["BB_Std"]
        g["BB_Lower"] = g["BB_Mid"] - 2 * g["BB_Std"]

        # Cumulative return
        g["Cumulative_Return"] = (1 + g["Daily_Return"].fillna(0)).cumprod() - 1

        out.append(g)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------
# 3. RISK / RETURN METRICS
# ------------------------------------------------------------------
def compute_risk_metrics(df):
    rows = []
    for ticker, g in df.groupby("Ticker"):
        daily_ret = g["Daily_Return"].dropna()
        ann_return = daily_ret.mean() * 252
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

        cum = (1 + daily_ret).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_drawdown = drawdown.min()

        total_return = g["Cumulative_Return"].iloc[-1]

        rows.append({
            "Ticker": ticker,
            "Sector": g["Sector"].iloc[0],
            "Total_Return_%": round(total_return * 100, 2),
            "Annualized_Return_%": round(ann_return * 100, 2),
            "Annualized_Volatility_%": round(ann_vol * 100, 2),
            "Sharpe_Ratio": round(sharpe, 2),
            "Max_Drawdown_%": round(max_drawdown * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("Sharpe_Ratio", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# 4. TREND CLASSIFICATION
# ------------------------------------------------------------------
def classify_trend(df):
    """Classify each stock's current trend based on price vs SMA50/SMA200
    (classic 'Golden Cross' / 'Death Cross' style logic)."""
    latest = df.sort_values("Date").groupby("Ticker").tail(1)
    trends = []
    for _, row in latest.iterrows():
        if row["SMA50"] > row["SMA200"] and row["Close"] > row["SMA50"]:
            trend = "Strong Uptrend"
        elif row["SMA50"] > row["SMA200"]:
            trend = "Mild Uptrend"
        elif row["SMA50"] < row["SMA200"] and row["Close"] < row["SMA50"]:
            trend = "Strong Downtrend"
        else:
            trend = "Sideways / Consolidating"
        trends.append({"Ticker": row["Ticker"], "Current_Trend": trend,
                        "RSI_14": round(row["RSI_14"], 1)})
    return pd.DataFrame(trends)


# ------------------------------------------------------------------
# 5. CHARTS
# ------------------------------------------------------------------
def make_charts(df, risk_df):
    tickers = sorted(df["Ticker"].unique())
    colors = sns.color_palette("deep", len(tickers))

    # --- Chart 1: Price trend with SMA20/50/200 for each stock (grid) ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.flatten()
    for i, ticker in enumerate(tickers):
        g = df[df["Ticker"] == ticker]
        ax = axes[i]
        ax.plot(g["Date"], g["Close"], label="Close", linewidth=1.2, color=colors[i])
        ax.plot(g["Date"], g["SMA50"], label="SMA50", linewidth=1, linestyle="--", color="orange")
        ax.plot(g["Date"], g["SMA200"], label="SMA200", linewidth=1, linestyle="--", color="red")
        ax.set_title(f"{ticker} ({g['Sector'].iloc[0]}) - Price & Moving Averages")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig("charts/01_price_trends_with_moving_averages.png", bbox_inches="tight")
    plt.close()

    # --- Chart 2: Cumulative returns comparison (all stocks, one chart) ---
    plt.figure(figsize=(12, 6))
    for i, ticker in enumerate(tickers):
        g = df[df["Ticker"] == ticker]
        plt.plot(g["Date"], g["Cumulative_Return"] * 100, label=ticker, color=colors[i])
    plt.title("Cumulative Return Comparison Across Stocks (%)")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/02_cumulative_returns_comparison.png", bbox_inches="tight")
    plt.close()

    # --- Chart 3: Rolling volatility comparison ---
    plt.figure(figsize=(12, 6))
    for i, ticker in enumerate(tickers):
        g = df[df["Ticker"] == ticker]
        plt.plot(g["Date"], g["Volatility_20D"] * 100, label=ticker, color=colors[i], linewidth=1)
    plt.title("20-Day Rolling Annualized Volatility (%)")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/03_rolling_volatility.png", bbox_inches="tight")
    plt.close()

    # --- Chart 4: Correlation heatmap of daily returns ---
    pivot = df.pivot(index="Date", columns="Ticker", values="Daily_Return")
    corr = pivot.corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
    plt.title("Correlation of Daily Returns Across Stocks")
    plt.tight_layout()
    plt.savefig("charts/04_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # --- Chart 5: Risk vs Return scatter ---
    plt.figure(figsize=(8, 6))
    for i, row in risk_df.iterrows():
        plt.scatter(row["Annualized_Volatility_%"], row["Annualized_Return_%"], s=140)
        plt.annotate(row["Ticker"],
                     (row["Annualized_Volatility_%"], row["Annualized_Return_%"]),
                     textcoords="offset points", xytext=(8, 5), fontsize=10)
    plt.xlabel("Annualized Volatility (Risk) %")
    plt.ylabel("Annualized Return %")
    plt.title("Risk vs Return by Stock")
    plt.tight_layout()
    plt.savefig("charts/05_risk_vs_return.png", bbox_inches="tight")
    plt.close()

    # --- Chart 6: RSI + MACD for the top performer (deep dive) ---
    top_ticker = risk_df.iloc[0]["Ticker"]
    g = df[df["Ticker"] == top_ticker]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1]})

    axes[0].plot(g["Date"], g["Close"], label="Close", color="black", linewidth=1)
    axes[0].plot(g["Date"], g["BB_Upper"], label="Upper Band", color="grey", linestyle="--", linewidth=0.8)
    axes[0].plot(g["Date"], g["BB_Lower"], label="Lower Band", color="grey", linestyle="--", linewidth=0.8)
    axes[0].fill_between(g["Date"], g["BB_Lower"], g["BB_Upper"], alpha=0.08, color="grey")
    axes[0].set_title(f"{top_ticker} - Bollinger Bands, RSI & MACD (Top Sharpe Ratio Stock)")
    axes[0].legend(fontsize=8)

    axes[1].plot(g["Date"], g["RSI_14"], color="purple", linewidth=1)
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("RSI (14)")

    axes[2].plot(g["Date"], g["MACD"], label="MACD", color="blue", linewidth=1)
    axes[2].plot(g["Date"], g["MACD_Signal"], label="Signal", color="orange", linewidth=1)
    axes[2].set_ylabel("MACD")
    axes[2].legend(fontsize=8)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig("charts/06_deep_dive_top_performer.png", bbox_inches="tight")
    plt.close()

    # --- Chart 7: Average trading volume by ticker ---
    plt.figure(figsize=(9, 5))
    vol_avg = df.groupby("Ticker")["Volume"].mean().sort_values(ascending=False)
    sns.barplot(x=vol_avg.index, y=vol_avg.values / 1e6, palette="deep")
    plt.ylabel("Average Daily Volume (Millions)")
    plt.title("Average Daily Trading Volume by Stock")
    plt.tight_layout()
    plt.savefig("charts/07_average_volume.png", bbox_inches="tight")
    plt.close()

    print("Saved 7 charts to /charts")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    df = add_indicators(df)
    risk_df = compute_risk_metrics(df)
    trend_df = classify_trend(df)

    print("=== RISK / RETURN METRICS ===")
    print(risk_df.to_string(index=False))
    print("\n=== CURRENT TREND CLASSIFICATION ===")
    print(trend_df.to_string(index=False))

    make_charts(df, risk_df)

    df.to_csv("data/stock_prices_clean_with_indicators.csv", index=False)
    risk_df.to_csv("data/risk_return_summary.csv", index=False)
    trend_df.to_csv("data/trend_classification.csv", index=False)
    print("\nSaved cleaned dataset + summaries to /data")
