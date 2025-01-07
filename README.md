# ID2223 "Project name"
This repository contains our final project of ID2223 2024/2025 at KTH.

## Authors
Erik Halme (ehalme@kth.se) \
Oskar Pålhagen (palhagen@kth.se)

## Technology stack
- Huggingface Transformers (finbert)
- Pygooglenews (Google News RSS feed)
- Yfinance (Yahoo! Finance API)
- Hopsworks (Model store)
- Pytorch (LSTM)




## Project Description
Forecasting of stock prices is crucial for investors, traders, and financial institutions to make informed decisions, manage risks, and capitalize on market opportunities.This project is a comprehensive Stock Price Predictor app designed to forecast the movements of the 30 most traded stocks on the Stockholm Stock Exchange (STO) by leveraging two key sources of data: historical stock prices, and sentiment analysis of news articles. The backbone of the prediction model is a Long Short-Term Memory (LSTM) neural network, which excels at identifying and modeling temporal dependencies in sequential data. By combining this with sentiment analysis, the application integrates quantitative and qualitative insights to provide more informed predictions.

### Architecture and dataset
In order to make predictions on stock prices we utlize both historical price data and news sentiment analysis to train an LSTM. Relevant news data is collected by scraping the Google News RSS feed using keywords through the [pygooglenews](https://pypi.org/project/pygooglenews/) package. The news data is fed to [finbert](https://arxiv.org/pdf/1908.10063) which returns sentiment information about the data. Historical price data is collected using Yahoo! Finance' API through the [yfinance](https://pypi.org/project/yfinance/) package. We utilize [Hopsworks](https://www.hopsworks.ai) as a model and feature store which feed into our training pipeline.

### Model
We utilize a Long Short-Term Memory (LSTM) neural network to predict stock prices. The LSTM is trained using the Adam optimizer and Mean Squared Error loss function. The LSTM is trained on a sequence of historical price data and news sentiment data to predict future price movements. The finall model uses a sequence of 30 data points to predict the next 5 data points. 

## Limitations 
Unfortanatly Googles RRS feed is limited to only include the latest year of news data. This means that we are limited to only using the latest year of news data for our predictions.
As news articles for some companies are sparse, we implemented a news decay function to account for this. We also tried to pretrain the modell on more price data and later fine tune it on combined price and news data this however did not improve the model performance.

## How to run the code
