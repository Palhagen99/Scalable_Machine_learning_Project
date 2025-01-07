import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import hopsworks
from tensorflow.keras.models import load_model
from DataLoader import DataLoader
from sklearn.preprocessing import MinMaxScaler

def predict(index_name="^OMX"):
    # Load the model
    project = hopsworks.login(api_key_value="pwWjyzF8SYsYJGQp.uZRknwAGCDPMe2covG1e4uVY4LsJXhAyKYgUNADOGY3H67mRAzoBtEJGlskTWE8h")
    mr = project.get_model_registry()
    model = mr.get_model("FinanceModel", version=10)
    saved_model_dir = model.download()
    print(saved_model_dir)
    model = load_model(saved_model_dir + "/model.keras")
    
    #Fetch the data used to train the model
    time_period_news = '30d'
    time_period_price = '3mo' #Needed to make sure we get 30 days of price data. Stock markets are closed on weekends and holidays
    data_loader = DataLoader(index_name, time_period_news, time_period_price)
    data = data_loader.get_data()

    #Get the previous closing price
    previous_closing_price = data['Close'].values
    #Remove uneccessary data and scale the data
    #The modell only takes the latest 30 days of data
    data = data[-30:]

    #Load the input and output scalers used to train the model
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    #Create a fake output data to fit the scaler
    output_scaler.fit_transform(previous_closing_price.reshape(-1, 1))

    #Scale the data
    data = input_scaler.fit_transform(data)

    #Format the data to be used by the model. The model expects the data to be in the shape (1, 30, 7)
    data = data.reshape(1, 30, 7)
    prediction = model.predict(data)

    #Inverse the scaling
    prediction = output_scaler.inverse_transform(prediction)[0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(previous_closing_price)), previous_closing_price, label="True Values", color="blue")
    predicted_indices = np.arange(len(previous_closing_price), len(previous_closing_price) + len(prediction))
    ax.scatter(predicted_indices, prediction, color="red", label="Predicted Value")
    ax.axvline(len(previous_closing_price) - 1, linestyle="--", color="gray", alpha=0.6)
    ax.set_title(f"Prediction for {index_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value")
    ax.legend()

    """ fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(previous_closing_price, label='Previous Closing Prices', linestyle='--',)

    # Create an array of indices for the predicted values, right after the last index of prev_closing
    predicted_indices = np.arange(len(previous_closing_price), len(previous_closing_price) + len(prediction))

    # Plot the predicted closing prices in red, using the correct indices
    ax.plot(predicted_indices, prediction, color='red', label='Predicted Prices',linestyle='--',) """
    
    return fig

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Financial Index Name", placeholder="Enter the name of the financial index..."),
    outputs=gr.Plot(label="Index Prediction Plot"),
    title="Financial Index Predictor",
    description="Enter the name of a financial index to generate a plot showing true values for the past 30 days and the predicted value."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
