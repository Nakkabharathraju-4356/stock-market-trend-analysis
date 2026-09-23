import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from run_analysis import (
    load_and_clean,
    add_indicators,
    compute_risk_metrics,
    classify_trend,
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Stock Market Trend Analysis",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📈 Stock Market Trend Analysis")
st.markdown(
    "Interactive analysis of stock trends, technical indicators, "
    "risk, returns and market relationships."
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def prepare_data():
    df = load_and_clean("data/stock_prices_raw.csv")
    df = add_indicators(df)

    risk_df = compute_risk_metrics(df)
    trend_df = classify_trend(df)

    return df, risk_df, trend_df


try:
    df, risk_df, trend_df = prepare_data()

except Exception as e:
    st.error("Unable to load the stock data.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Dashboard Controls")

tickers = sorted(df["Ticker"].unique())

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    tickers
)

# ---------------------------------------------------------
# SELECTED STOCK DATA
# ---------------------------------------------------------

stock_df = df[df["Ticker"] == selected_ticker].copy()
stock_df = stock_df.sort_values("Date")

latest = stock_df.iloc[-1]

risk_row = risk_df[
    risk_df["Ticker"] == selected_ticker
]

trend_row = trend_df[
    trend_df["Ticker"] == selected_ticker
]

# ---------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------

st.subheader(f"{selected_ticker} Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"{latest['Close']:.2f}"
    )

with col2:
    st.metric(
        "RSI (14)",
        f"{latest['RSI_14']:.2f}"
    )

with col3:
    st.metric(
        "Total Return",
        f"{risk_row.iloc[0]['Total_Return_%']:.2f}%"
    )

with col4:
    st.metric(
        "Volatility",
        f"{risk_row.iloc[0]['Annualized_Volatility_%']:.2f}%"
    )

with col5:
    st.metric(
        "Sharpe Ratio",
        f"{risk_row.iloc[0]['Sharpe_Ratio']:.2f}"
    )

# ---------------------------------------------------------
# TREND
# ---------------------------------------------------------

st.subheader("Current Trend")

trend = trend_row.iloc[0]["Current_Trend"]

if trend == "Strong Uptrend":
    st.success(f"📈 {trend}")

elif trend == "Mild Uptrend":
    st.info(f"📈 {trend}")

elif trend == "Strong Downtrend":
    st.error(f"📉 {trend}")

else:
    st.warning(f"↔️ {trend}")

# ---------------------------------------------------------
# PRICE + MOVING AVERAGES
# ---------------------------------------------------------

st.subheader("📊 Price & Moving Averages")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    stock_df["Date"],
    stock_df["Close"],
    label="Close Price",
    linewidth=1.5
)

ax.plot(
    stock_df["Date"],
    stock_df["SMA20"],
    label="SMA20",
    linestyle="--"
)

ax.plot(
    stock_df["Date"],
    stock_df["SMA50"],
    label="SMA50",
    linestyle="--"
)

ax.plot(
    stock_df["Date"],
    stock_df["SMA200"],
    label="SMA200",
    linestyle="--"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ---------------------------------------------------------
# CUMULATIVE RETURN
# ---------------------------------------------------------

st.subheader("📈 Cumulative Return")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    stock_df["Date"],
    stock_df["Cumulative_Return"] * 100,
    linewidth=1.5
)

ax.axhline(0, linestyle="--")

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (%)")
ax.grid(True, alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ---------------------------------------------------------
# RSI
# ---------------------------------------------------------

st.subheader("📉 RSI (14)")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    stock_df["Date"],
    stock_df["RSI_14"],
    linewidth=1.3
)

ax.axhline(70, linestyle="--")
ax.axhline(30, linestyle="--")

ax.set_ylim(0, 100)
ax.set_xlabel("Date")
ax.set_ylabel("RSI")
ax.grid(True, alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ---------------------------------------------------------
# MACD
# ---------------------------------------------------------

st.subheader("📊 MACD")

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    stock_df["Date"],
    stock_df["MACD"],
    label="MACD"
)

ax.plot(
    stock_df["Date"],
    stock_df["MACD_Signal"],
    label="Signal"
)

ax.axhline(0, linestyle="--")

ax.set_xlabel("Date")
ax.set_ylabel("MACD")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ---------------------------------------------------------
# RISK / RETURN
# ---------------------------------------------------------

st.subheader("⚖️ Risk & Return Analysis")

selected_risk = risk_row[
    [
        "Ticker",
        "Sector",
        "Total_Return_%",
        "Annualized_Return_%",
        "Annualized_Volatility_%",
        "Sharpe_Ratio",
        "Max_Drawdown_%"
    ]
]

st.dataframe(
    selected_risk,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# ALL STOCKS COMPARISON
# ---------------------------------------------------------

st.subheader("📋 All Stocks — Risk & Return Summary")

st.dataframe(
    risk_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# TREND CLASSIFICATION
# ---------------------------------------------------------

st.subheader("📊 Current Trend Classification")

st.dataframe(
    trend_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------

st.subheader("🔥 Stock Return Correlation")

pivot = df.pivot(
    index="Date",
    columns="Ticker",
    values="Daily_Return"
)

corr = pivot.corr()

fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True,
    ax=ax
)

ax.set_title("Correlation of Daily Returns")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ---------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------

with st.expander("🔍 View Stock Data"):

    st.dataframe(
        stock_df,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Stock Market Trend Analysis | Python | Pandas | NumPy | "
    "Matplotlib | Seaborn | Streamlit"
)
