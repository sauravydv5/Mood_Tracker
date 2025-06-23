# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment_model import analyze_text

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    result = analyze_text(text)
    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
