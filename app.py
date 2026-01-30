
import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

st.title("ARIMA Stock Forecast")

ticker = st.text_input("Ticker", "AAPL")
period = st.selectbox("Period", ["1y", "2y", "5y"])

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    return df

if st.button("Run Forecast"):
    data = load_data(ticker, period)

    if data.empty or "Close" not in data:
        st.error("Failed to load data (Yahoo Finance rate-limited). Try again later.")
        st.stop()

    close = data["Close"].dropna()

    if len(close) < 50:
        st.error("Not enough data points for ARIMA.")
        st.stop()

    # Force a business-day frequency
    close.index = pd.to_datetime(close.index)
    close = close.asfreq("B")
    close = close.fillna(method="ffill")

    st.subheader("Historical Prices")
    st.line_chart(close)

    try:
        model = ARIMA(close, order=(1,1,1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=30)

        st.subheader("30-Day Forecast")
        st.line_chart(forecast)

    except Exception as e:
        st.error("Model failed to converge.")
        st.code(str(e))