#itterativly use the provided model to forecast the stock price ahead of time
#Data will be a pandas dataframe with the same columns as the data used to train the model
#All data will need to use a Scaler
#Forecast period will be the number of days ahead to forecast
#The function will return a pandas dataframe with the forecasted stock prices normalized
import hopsworks


def forecast_stock_price(model=None, data, forecast_period, Model_from_hopswork=False):
    #if model from hopswork is true, the model will be downloaded from the hopswork model registry
    if Model_from_hopswork:
        hopsworks.login()






