import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import datetime
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

def fetch_stock_data_from_exchanges(ticker, start_date, end_date):
    exchanges = ['.NS', '.BO']
    for exchange in exchanges:
        full_ticker = ticker if ticker.endswith('.NS') or ticker.endswith('.BO') else ticker + exchange

        try:
            stock_data = yf.download(full_ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                return stock_data, full_ticker
        except Exception as e:
            print(f"Error fetching data for {ticker} on {exchange[1:]}: {e}")
    return None, None

def fetch_news_sentiment(ticker):
    news_url = f"https://news.google.com/search?q={ticker} stock"
    response = requests.get(news_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [headline.text for headline in soup.find_all('h3')]
    
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [sentiment_analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    
    return average_sentiment

def calculate_technical_indicators(stock_data):
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()

    delta = stock_data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    stock_data['MiddleBand'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['UpperBand'] = stock_data['MiddleBand'] + 1.96 * stock_data['Close'].rolling(window=20).std()
    stock_data['LowerBand'] = stock_data['MiddleBand'] - 1.96 * stock_data['Close'].rolling(window=20).std()

    stock_data = stock_data.dropna()

    return stock_data

def prepare_data(stock_data, sentiment_score):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']])

    sentiment_feature = np.full((scaled_data.shape[0], 1), sentiment_score)
    extended_data = np.hstack([scaled_data, sentiment_feature])

    X, y = [], []
    for i in range(5, len(extended_data)):
        X.append(extended_data[i-5:i, :])
        y.append(extended_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
    return model, history

def predict_stock_prices(stock_data, model, scaler, sentiment_score):
    scaled_data = scaler.transform(stock_data[['Close']])
    sentiment_feature = np.full((scaled_data.shape[0], 1), sentiment_score)
    extended_data = np.hstack([scaled_data, sentiment_feature])

    X_test = []
    for i in range(5, len(extended_data)):
        X_test.append(extended_data[i-5:i, :])
    X_test = np.array(X_test)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    next_day_prediction = predictions[-1][0]
    print(f"Predicted closing price for the next trading day: {next_day_prediction}")
    
    return next_day_prediction

def plot_stock_data(stock_data, ticker, next_day_prediction=None):
    last_5_days_data = stock_data.tail(5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(last_5_days_data.index, last_5_days_data['Close'], color='blue', label=f'{ticker} Closing Prices')
    plt.scatter(last_5_days_data.index[-1], last_5_days_data['Close'].iloc[-1], color='blue', label='Last Closing Price')

    if next_day_prediction is not None:
        plt.scatter(last_5_days_data.index[-1] + pd.Timedelta(days=1), next_day_prediction, color='red', label='Next Day Prediction')

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'{ticker} Stock Price - Last 5 Days and Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter the stock ticker symbol (without exchange suffix): ").upper()
    
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    end_date = datetime.datetime.now()
    
    stock_data, full_ticker = fetch_stock_data_from_exchanges(ticker, start_date, end_date)
    
    if stock_data is not None:
        sentiment_score = fetch_news_sentiment(ticker)
        print(f"Average sentiment score for news: {sentiment_score}")

        stock_data = calculate_technical_indicators(stock_data)

        X, y, scaler = prepare_data(stock_data, sentiment_score)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        model, history = train_lstm_model(X, y)
        
        next_day_prediction = predict_stock_prices(stock_data, model, scaler, sentiment_score)
        
        plot_stock_data(stock_data, full_ticker, next_day_prediction)
        
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        mse = mean_squared_error(stock_data['Close'].values[5:], y_pred)
        print(f"Mean Squared Error: {mse}")
    else:
        print("Stock data not found on NSE or BSE.")
