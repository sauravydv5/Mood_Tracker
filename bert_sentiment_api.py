# sentiment_model.py
from transformers import pipeline

# Load the Hugging Face sentiment pipeline once
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_text(text):
    if not text:
        return {"error": "Text is required"}

    result = sentiment_pipeline(text)[0]
    mood = result['label'].capitalize()
    score = round(result['score'], 4)

    return {
        "mood": mood,
        "confidence": score
    }
