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

def get_ticker(stock_name):
    return stock_name.upper()

def get_historical_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def get_news_sentiment(news_headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
    return np.mean(sentiments) if sentiments else 0

def train_model(data):
    data['Prediction'] = data['Adj Close'].shift(-1)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Prediction'].dropna()
    X = X[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model MSE: {mse}")

    return model

def predict_future(model, recent_data, news_sentiment):
    prediction_input = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(prediction_input)
    adjusted_prediction = prediction[0] * (1 + news_sentiment)
    return adjusted_prediction

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Price Prediction Tool"),
    dcc.Input(id='stock-input', type='text', placeholder='Enter Stock Ticker (e.g., YESBANK.NS)', value='YESBANK.NS'),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
        end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
    ),
    html.Button('Predict', id='predict-button'),
    dcc.Graph(id='price-graph'),
])

@app.callback(
    Output('price-graph', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [Input('stock-input', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(n_clicks, stock_name, start_date, end_date):
    if not stock_name or not start_date or not end_date:
        return {}

    ticker = get_ticker(stock_name)
    data = get_historical_data(ticker, start_date, end_date)

    if data.empty:
        return {}

    news_headlines = ["Positive earnings report", "New product launch", "Stock buyback announcement"]
    news_sentiment = get_news_sentiment(news_headlines)

    model = train_model(data)
    future_price = predict_future(model, data, news_sentiment)
    future_date = data.index[-1] + pd.DateOffset(1)

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Historical Prices'))

    price_fig.add_trace(go.Scatter(
        x=[data.index[-1], future_date],
        y=[data['Adj Close'].iloc[-1], future_price],
        mode='lines+markers',
        name='Prediction',
        line=dict(color='red'),
        marker=dict(size=8, color='red')
    ))

    return price_fig

if __name__ == '__main__':
    app.run_server(debug=True)
