import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

ticker = st.text_input("Enter stock ticker", "AAPL")
period = st.selectbox("Select period", ["1y", "2y", "5y"])

if st.button("Run Prediction"):
    data = yf.download(ticker, period=period)
    close = data["Close"].dropna()

    model = ARIMA(close, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)

    st.subheader("Last prices")
    st.line_chart(close)

    st.subheader("30-Day Forecast")
    st.line_chart(forecast)