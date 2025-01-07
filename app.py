import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import hopsworks
from tensorflow.keras.models import load_model
from DataLoader import DataLoader
from sklearn.preprocessing import MinMaxScaler

name_to_ticker = {
    "ABB Ltd": "ABB.ST",
    "Alfa Laval": "ALFA.ST",
    "Assa Abloy B": "ASSA-B.ST",
    "Astra Zeneca": "AZN.ST",
    "Atlas Copco A": "ATCO-A.ST",
    "Atlas Copco B": "ATCO-B.ST",
    "Boliden": "BOL.ST",
    "Electrolux B": "ELUX-B.ST",
    "Ericsson B": "ERIC-B.ST",
    "Essity B": "ESSITY-B.ST",
    "Evolution": "EVO.ST",
    "Getinge B": "GETI-B.ST",
    "Hennes & Mauritz B": "HM-B.ST",
    "Hexagon AB": "HEXA-B.ST",
    "Investor B": "INVE-B.ST",
    "Kinnevik B": "KINV-B.ST",
    "Nordea Bank Abp": "NDA-SE.ST",
    "Sandvik": "SAND.ST",
    "Sinch B": "SINCH.ST",
    "SEB A": "SEB-A.ST",
    "Skanska B": "SKA-B.ST",
    "SKF B": "SKF-B.ST",
    "SCA B": "SCA-B.ST",
    "Svenska Handelsbanken A": "SHB-A.ST",
    "Swedbank A": "SWED-A.ST",
    "Tele2 B": "TEL2-B.ST",
    "Telia": "TELIA.ST",
    "Volvo B": "VOLV-B.ST",
}

def predict(stock_name):
    ticker = name_to_ticker[stock_name]
    print(f"TICKER VALUE: {ticker}")
    # Load the model
    project = hopsworks.login(
        api_key_value="pwWjyzF8SYsYJGQp.uZRknwAGCDPMe2covG1e4uVY4LsJXhAyKYgUNADOGY3H67mRAzoBtEJGlskTWE8h"
    )
    mr = project.get_model_registry()
    model = mr.get_model("FinanceModel", version=11)
    saved_model_dir = model.download()
    print(saved_model_dir)
    model = load_model(saved_model_dir + "/model.keras")

    # Fetch the data used to train the model
    time_period_news = "30d"
    time_period_price = "3mo"  # Needed to make sure we get 30 days of price data. Stock markets are closed on weekends and holidays
    data_loader = DataLoader(ticker, time_period_news, time_period_price)
    data = data_loader.get_data()

    # Get the previous closing price
    previous_closing_prices = data["Close"].values

    data = data[-30:]

    # Load the input and output scalers used to train the model
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    # Scale output to specific stock
    output_scaler.fit_transform(previous_closing_prices.reshape(-1, 1))

    # Scale the data
    data = input_scaler.fit_transform(data)

    # Format the data to be used by the model. The model expects the data to be in the shape (1, 30, 7)
    data = data.reshape(1, 30, 8)
    prediction = model.predict(data)
    print(f"PREDICTION: {prediction}")

    # Inverse the scaling
    prediction = output_scaler.inverse_transform(prediction)[0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    previous_indices = range(len(previous_closing_prices))
    ax.plot(
        previous_indices,
        previous_closing_prices,
        label="True Values",
        color="blue",
    )
    predicted_indices = np.arange(
        len(previous_closing_prices), len(previous_closing_prices) + len(prediction)
    )
    ax.plot([previous_indices[-1], predicted_indices[0]], [previous_closing_prices[-1], prediction[0]], color="red")
    ax.plot(predicted_indices, prediction, color="red", label="Predicted Values")
    ax.axvline(len(previous_closing_prices) - 1, linestyle="--", color="gray", alpha=0.6)
    ax.set_title(f"Prediction for {stock_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    return fig

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(
        label="Stock name",
        choices=list(name_to_ticker.keys()),
        value="ABB.ST",  # Default value
    ),
    outputs=gr.Plot(label="Index Prediction Plot"),
    title="Financial Index Predictor",
    description="Enter the name of a financial index to generate a plot showing true values for the past 30 days and the predicted value.",
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
