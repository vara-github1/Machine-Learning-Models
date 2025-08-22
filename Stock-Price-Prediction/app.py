import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import os

st.sidebar.title('Stock Price Prediction App')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

# Function to download stock data
@st.cache_resource
def download_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date, progress=False)

# Sidebar inputs
ticker = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL').upper()
today = date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value=3000)
start_date = st.sidebar.date_input('Start Date', value=today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', value=today)

if st.sidebar.button('Download Data'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: {start_date}\nEnd date: {end_date}')
        df = download_data(ticker, start_date, end_date)
        
        # Ensure 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        file_path = f"data/{ticker}.csv"
        df.to_csv(file_path)
        st.sidebar.success(f"Data saved successfully as {file_path}")
    else:
        st.sidebar.error('Error: End date must be after start date')

# Load data
data = download_data(ticker, start_date, end_date)
scaler = StandardScaler()

def visualize_data(ticker):
    st.header(f"{ticker}'s Stock Trends Visualization")

    st.subheader('Closing Price vs Time Chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Closing Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Closing Price Over Time')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
    ma100 = data['Close'].rolling(100).mean()
    ma200 = data['Close'].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Closing Price', color='blue', linewidth=1)
    ax.plot(ma100, label='100-day MA', color='red', linestyle='dashed', linewidth=1.5)
    ax.plot(ma200, label='200-day MA', color='green', linestyle='dashed', linewidth=1.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Closing Price with 100MA & 200MA')
    ax.legend(loc='upper left', fontsize='medium', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


def recent_data(ticker):

    st.header(f'Recent Data of {ticker}')
    st.dataframe(data.tail(100))

def predict():
    st.header('Stock Price predictions')

    model_choice = st.radio('Choose a model', ['XGBoostRegressor', 'LSTM', 'ARIMA', 'Prophet'])
    num_days = st.number_input('How many days to forecast?', value=3, min_value=1, step=1)

    if st.button('Predict'):
        if model_choice == 'XGBoostRegressor':
            xgboost_model(num_days)
        elif model_choice == 'LSTM':
            lstm_model(num_days)
        elif model_choice == 'ARIMA':
            arima_model(num_days)
        else:
            prophet_model(num_days)

def xgboost_model(num_days):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num_days)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    y = df.preds.values[:-num_days]
    x_train, x_test, y_train, y_test = train_test_split(x[:-num_days], y, test_size=0.2, random_state=7)
    
    model = XGBRegressor()
    model.fit(x_train, y_train)
    
    preds = model.predict(x_test)
    # st.text(f'R² Score: {r2_score(y_test, preds)}\nMAE: {mean_absolute_error(y_test, preds)}')
    
    future_forecast = model.predict(x[-num_days:])
    for i, val in enumerate(future_forecast, 1):
        st.text(f'Day {i}: {val}')

def lstm_model(num_days):
    df = data[['Close']]
    if df.empty:
        st.error("No data available. Try another stock symbol or date range.")
        return
    
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(30, len(scaled_data) - num_days):
        X.append(scaled_data[i-30:i])
        y.append(scaled_data[i])
    
    if len(X) == 0 or len(y) == 0:
        st.error("Not enough data points for LSTM. Try selecting a longer duration.")
        return
    
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(30, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    forecast_input = scaled_data[-30:].reshape(1, 30, 1)
    predictions = [scaler.inverse_transform(model.predict(forecast_input))[0][0] for _ in range(num_days)]
    
    for i, val in enumerate(predictions, 1):
        st.text(f'Day {i}: {val}')

def arima_model(num_days):
    df = data[['Close']]
    model = ARIMA(df, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=num_days)
    
    for i, val in enumerate(forecast, 1):
        st.text(f'Day {i}: {val}')

def prophet_model(num_days):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=num_days)
    forecast = model.predict(future)
    predictions = forecast[['ds', 'yhat']].tail(num_days)
    for i, row in predictions.iterrows():
        st.text(f'{row.ds.date()}: {row.yhat}')

def main():
    option = st.sidebar.selectbox('Choose an option', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        visualize_data(ticker)
    elif option == 'Recent Data':
        recent_data(ticker)
    else:
        predict()

if __name__ == '__main__':
    main()
