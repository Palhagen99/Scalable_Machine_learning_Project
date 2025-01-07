# ID2223 "Project name"
This repository contains our final project of ID2223 2024/2025 at KTH.

## Authors
Erik Halme (ehalme@kth.se) \
Oskar Pålhagen (palhagen@kth.se)

## Project Description
Forecasting of stock and index prices is crucial for investors, traders, and financial institutions to make informed decisions, manage risks, and capitalize on market opportunities.This project is a comprehensive Index and Stock Price Predictor app designed to forecast financial market movements by leveraging two key sources of data: historical stock and index prices, and sentiment analysis of news articles. The backbone of the prediction model is a Long Short-Term Memory (LSTM) neural network, which excels at identifying and modeling temporal dependencies in sequential data. By combining this with sentiment analysis, the application integrates quantitative and qualitative insights to provide more informed predictions.

### Architecture and dataset
In order to make predictions on index and stock prices we utlize both historical price data and news sentiment analysis to train an LSTM. Relevant news data is collected by scraping the Google News RSS feed using keywords through the [pygooglenews](https://pypi.org/project/pygooglenews/) package. The news data is fed to [finbert](https://arxiv.org/pdf/1908.10063) which returns sentiment information about the data. Historical price data is collected using Yahoo! Finance' API through the [yfinance](https://pypi.org/project/yfinance/) package. We utilize [Hopsworks](https://www.hopsworks.ai) as a model and feature store which feed into our training pipeline.

## How to run the code
