from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Force using PyTorch backend to avoid Keras issues
sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")

@app.route('/x', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    result = sentiment_pipeline(text)[0]
    mood = result['label'].capitalize()
    score = round(result['score'], 4)

    return jsonify({
        "mood": mood,
        "confidence": score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
