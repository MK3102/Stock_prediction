import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import datetime

# Convert stock name to ticker symbol
def get_ticker(stock_name):
    nse_stocks = {
        'RELIANCE': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'INFY': 'INFY.NS',
        # Add more NSE stock mappings here
    }
    bse_stocks = {
        'RELIANCE': '500325.BO',
        'TCS': '532540.BO',
        'INFY': '500209.BO',
        # Add more BSE stock mappings here
    }
    
    if stock_name.upper() in nse_stocks:
        return nse_stocks[stock_name.upper()]
    elif stock_name.upper() in bse_stocks:
        return bse_stocks[stock_name.upper()]
    else:
        raise ValueError("Stock name not found in NSE or BSE listings.")

# Download historical data
def get_historical_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# News Sentiment Analysis
def get_news_sentiment(news_headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for headline in news_headlines:
        sentiment = analyzer.polarity_scores(headline)
        sentiments.append(sentiment['compound'])
    return np.mean(sentiments)

# Train ML model
def train_model(data):
    data['Prediction'] = data['Adj Close'].shift(-1)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Prediction'].dropna()

    X = X[:-1]  # align with y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model MSE: {mse}")

    return model, X_test, y_test

# Predict future prices
def predict_future(model, recent_data, news_sentiment):
    prediction_input = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(prediction_input)
    adjusted_prediction = prediction * (1 + news_sentiment)
    return adjusted_prediction

# Visualization using Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Indian Stock Price Prediction Tool"),
    dcc.Input(id='stock-input', type='text', placeholder='Enter Stock Name (e.g., RELIANCE)', value='RELIANCE'),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
        end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
    ),
    html.Button('Predict', id='predict-button'),
    dcc.Graph(id='price-graph'),
    dcc.Graph(id='prediction-graph'),
])

@app.callback(
    [Output('price-graph', 'figure'),
     Output('prediction-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [Input('stock-input', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(n_clicks, stock_name, start_date, end_date):
    if not stock_name or not start_date or not end_date:
        return {}, {}

    try:
        ticker = get_ticker(stock_name)
    except ValueError as e:
        return {}, {}

    # Fetch historical data
    data = get_historical_data(ticker, start_date, end_date)

    # Sample news headlines for sentiment analysis
    news_headlines = ["Positive earnings report", "New product launch", "Stock buyback announcement"]
    news_sentiment = get_news_sentiment(news_headlines)

    # Train model and predict future prices
    model, X_test, y_test = train_model(data)
    future_price = predict_future(model, data, news_sentiment)

    # Visualize historical data
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical Prices'))

    # Visualize prediction
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual Prices'))
    prediction_fig.add_trace(go.Scatter(x=X_test.index, y=model.predict(X_test), mode='lines', name='Predicted Prices'))
    prediction_fig.add_trace(go.Scatter(x=[data.index[-1] + pd.DateOffset(1)], y=[future_price], mode='markers', name='Future Prediction'))

    return price_fig, prediction_fig

if __name__ == '__main__':
    app.run_server(debug=True)
