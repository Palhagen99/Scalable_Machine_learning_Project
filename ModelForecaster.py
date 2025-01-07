#itterativly use the provided model to forecast the stock price ahead of time
#Data will be a pandas dataframe with the same columns as the data used to train the model
#All data will need to use a Scaler
#Forecast period will be the number of days ahead to forecast
#The function will return a pandas dataframe with the forecasted stock prices normalized

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from DataLoader import DataLoader
import joblib




def forecast_stock_price(ticker):
    """
    Function to forecast the stock price of a given ticker


    :param stockticker:
    :returns: prediction, previous_closing_price

    """
    model_dir = "./Resources/Models"
    model_path = model_dir + "/model.keras"
    model = load_model(model_path)

    #Fetch the data used to train the model
    time_period_news = '30d'
    time_period_price = '3mo' #Needed to make sure we get 30 days of price data. Stock markets are closed on weekends and holidays
    data_loader = DataLoader(ticker, time_period_news, time_period_price)
    data = data_loader.get_data()

    #Get the previous closing price
    previous_closing_price = data['Close'].values
    #Remove uneccessary data and scale the data
    #The modell only takes the latest 30 days of data
    data = data[-30:]


    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    #Use latest 30 days to create a scaler for the input data wich will be used to invert the scaling of the output
    output_scaler.fit_transform(previous_closing_price.reshape(-1, 1))

    #Scale the data
    data = input_scaler.fit_transform(data)

    #Format the data to be used by the model. The model expects the data to be in the shape (1, 30, 8)
    data = data.reshape(1, 30, 8)

    prediction = model.predict(data)


    #Inverse the scaling
    prediction = output_scaler.inverse_transform(prediction)

    return prediction[0], previous_closing_price

















