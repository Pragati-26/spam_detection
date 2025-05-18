# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load model and tokenizer
model = load_model("spam_model.h5")  # Use .h5 for Keras models
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Set the maxlen used during training
MAX_LEN = 100  # Replace with the actual maxlen you used during training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    # Tokenize and pad the input message
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    # Predict using the model
    prediction = model.predict(padded)
    result = "Spam" if prediction[0][0] >= 0.5 else "Not Spam"

    return jsonify({"prediction": result})

@app.route('/')
def home():
    return "Spam Detection API is running. Use /predict with POST."

if __name__ == '__main__':
    app.run(debug=True)
