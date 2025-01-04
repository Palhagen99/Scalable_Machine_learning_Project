import gradio as gr
from transformers import pipeline
import yfinance as yf
import matplotlib.pyplot as plt
#from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Function to generate a sine wave plot
def predict(index_name="^OMX"):
    index = yf.Ticker(index_name)
    index_data = index.history(period="3mo")
    index_data = index_data.reset_index()
    index_close_price_list = index_data.tail(30)["Close"].to_list()
    
    # Sentiment analysis using finbert
    news_data = "PLACEHOLDER"
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    sentiment_score = pipe(news_data, top_k=None)

    # Test random predicted value
    predicted_value = index_close_price_list[-1] + 10

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(index_close_price_list)), index_close_price_list, label="True Values", color="blue")
    ax.scatter(len(index_close_price_list), predicted_value, color="red", label="Predicted Value")
    ax.axvline(len(index_close_price_list) - 1, linestyle="--", color="gray", alpha=0.6)
    ax.set_title(f"Prediction for {index_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value")
    ax.legend()
    
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
