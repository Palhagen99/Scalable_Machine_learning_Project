#Class to fetch news and stock data from the web for a specific ticker and combine them into a dataframe.

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from pygooglenews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
class DataLoader:
    def __init__(self, ticker, time_period_news, time_period_stock, news_decay_rate = 0):
        self.ticker = ticker
        self.time_period_news = time_period_news
        self.time_period_stock = time_period_stock
        self.news_decay_rate = news_decay_rate

    def get_data(self):
        stock_data = self.get_stock_data()
        news_data = self.get_news_data()
        news_sentiment = self.get_sentiment(news_data)
        combined_data = self.combine_data(stock_data, news_sentiment)

        if self.news_decay_rate != 0:
            combined_data = self.news_decay(combined_data, self.news_decay_rate)

        return combined_data


    def get_stock_data(self):
        data = yf.download(self.ticker, period = self.time_period_stock)
        df = pd.DataFrame()
        df['Open'] = data['Open']
        df['Close'] = data['Close']
        df['High'] = data['High']
        df['Low'] = data['Low']

        return df

    def get_news_data(self):
        googlenews = GoogleNews()
        news_data = googlenews.search(self.ticker, when=self.time_period_news)
        news_data = pd.DataFrame(news_data['entries'])
        return news_data

    def get_sentiment(self, news_data):
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

        news_sentiment = []
        for i in range(len(news_data)):
            sentiment = classifier(news_data['title'][i], top_k=None)
            postive_score = sentiment[0]['score']
            negative_score = sentiment[1]['score']
            neutral_score = sentiment[2]['score']
            reformmated_time_stamp = pd.to_datetime(news_data['published'][i]).date()
            news_sentiment.append({'Date': reformmated_time_stamp, 'positive_score': postive_score, 'negative_score': negative_score, 'neutral_score': neutral_score})
        return pd.DataFrame(news_sentiment)

    def combine_data(self, stock_data, news_sentiment):
        news_sentiment = (
            news_sentiment
            .groupby('Date')
            .mean()
            .fillna(0)
            .reset_index()
            .set_index('Date')
            .sort_index()
        )

        common_index = pd.date_range(
            start=pd.Timestamp(min(pd.Timestamp(stock_data.index[0]), pd.Timestamp(news_sentiment.index[0]))),
            end=pd.Timestamp(max(pd.Timestamp(stock_data.index[-1]), pd.Timestamp(news_sentiment.index[-1]))),
            freq='D'
        )
        stock_data = stock_data.reindex(common_index).fillna(-1)

        news_sentiment = news_sentiment.reindex(common_index).fillna(0)

        #Ensure stock_data and news_sentiment have combatile indices
        stock_data.index = pd.to_datetime(stock_data.index).normalize()
        news_sentiment.index = pd.to_datetime(news_sentiment.index).normalize()

        combined_data = pd.merge(
            stock_data,
            news_sentiment,
            how='left',
            left_index=True,
            right_index=True
        )

        #Drop all close values that are -1
        combined_data = combined_data[combined_data['Close'] != -1]
        #fill all missing values with 0
        combined_data = combined_data.fillna(0)

        return combined_data

    def news_decay(self, Combined_data, decay_rate):
        #We have lots of days in the data with no news. We will fill these days with the previous days news * decay_rate
        #This will allow us to have a more continuous news data
        combined_data = Combined_data.copy()
        news_columns = ['positive_score', 'negative_score', 'neutral_score']
        #We want to start from the oldest date and work our way to the newest date
        for i in range(1, len(combined_data)):
            for column in news_columns:
                if combined_data[column][i] == 0:
                    combined_data[column][i] = combined_data[column][i-1] * decay_rate
        return combined_data








