from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import gradio as gr
import numpy as np
import re
import pickle

# Load pre-trained tokenizer (assumes you have a saved tokenizer)
with open('/home/uday/Documents/projects/deployment/twitter_sentiment_analysis/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained sentiment analysis model
model = load_model('/home/uday/Documents/projects/deployment/twitter_sentiment_analysis/twitter_sentiment_analysis.keras')

# Labels for the model output
sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Make sure these match your model!

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    text_to_seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(text_to_seq, maxlen=50, padding='post', truncating='post')
    pred = model.predict(padded)
    if pred[0][1] > 0.6:
        predicted_class = 1  # Neutral
    else:
        predicted_class = np.argmax(pred)
    prediction = sentiment_labels[predicted_class]
    return f"Sentiment: {prediction}"

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a Tweet here..."),
    outputs="text",
    title="Twitter Sentiment Analysis",
    description="Paste a tweet to analyze its sentiment (positive, neutral, or negative)."
)

iface.launch()
