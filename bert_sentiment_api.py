# bert_sentiment_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the Hugging Face sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    result = sentiment_pipeline(text)[0]  # Result: {'label': 'NEGATIVE', 'score': 0.99}
    mood = result['label'].capitalize()
    score = round(result['score'], 4)

    return jsonify({
        "mood": mood,
        "confidence": score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)