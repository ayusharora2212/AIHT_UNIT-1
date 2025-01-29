from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('sentiment_model.pkl')

# Load the tokenizer (you need to save and reload your tokenizer used during training)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Label mapping dictionary
label_mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}

# Preprocess input text


def preprocess_input_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize and pad the text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=100)

    return text_padded


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the frontend
        input_data = request.get_json()
        text = input_data.get("text", "")

        # Preprocess the text
        processed_text = preprocess_input_text(text)

        # Make predictions
        prediction = model.predict(processed_text)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_sentiment = label_mapping[predicted_label]

        # Return the result as JSON
        return jsonify({
            "text": text,
            "sentiment": predicted_sentiment,
            # Include confidence scores for all classes
            "confidence": prediction[0].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
