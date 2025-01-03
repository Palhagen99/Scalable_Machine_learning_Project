import gradio as gr
from transformers import pipeline
import yfinance as yf
import pandas as pd
#from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Function to generate a sine wave plot
def predict(text):
    omx30 = yf.Ticker("^OMX")
    omx30_data = omx30.history(period="3mo")
    omx30_data = omx30_data.reset_index()
    omx30_data = omx30_data.tail(30)
    print(omx30_data)

    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return pipe(text, top_k=None)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
