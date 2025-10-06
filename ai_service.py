# ai_service.py - BERT-based Sentiment Analysis API
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# MODEL INITIALIZATION
# ============================================

class MentalHealthBERTAnalyzer:
    def __init__(self):
        logger.info("Loading BERT models...")
        
        # Option 1: General sentiment analysis (faster, lighter)
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Option 2: Emotion detection (more nuanced)
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None  # Return all emotions with scores
        )
        
        logger.info("Models loaded successfully!")
        logger.info(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using BERT
        Returns: sentiment (positive/negative/neutral) and confidence
        """
        try:
            result = self.sentiment_model(text)[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Map to our sentiment categories
            if label == 'positive':
                sentiment = 'positive'
            elif label == 'negative':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'raw_label': label
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def detect_emotions(self, text):
        """
        Detect multiple emotions from text
        Returns: List of emotions with confidence scores
        """
        try:
            results = self.emotion_model(text)[0]
            
            # Sort by score and get top emotions
            emotions = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Map emotion labels to our categories
            emotion_mapping = {
                'joy': 'happy',
                'sadness': 'sad',
                'anger': 'angry',
                'fear': 'anxious',
                'surprise': 'surprised',
                'disgust': 'disgusted',
                'neutral': 'neutral'
            }
            
            detected_emotions = []
            for emotion in emotions[:3]:  # Top 3 emotions
                mapped_emotion = emotion_mapping.get(emotion['label'].lower(), emotion['label'])
                detected_emotions.append({
                    'emotion': mapped_emotion,
                    'confidence': float(emotion['score']),
                    'original_label': emotion['label']
                })
            
            return detected_emotions
        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return []
    
    def assess_risk_level(self, text, emotions, sentiment_score):
        """
        Assess mental health risk level based on text analysis
        This is a placeholder - you'll add your psychological model here tomorrow
        """
        # Keywords that might indicate distress
        high_risk_keywords = [
            'suicide', 'kill myself', 'end it all', 'no point', 'give up',
            'hopeless', 'worthless', 'hate myself', 'want to die'
        ]
        
        medium_risk_keywords = [
            'depressed', 'anxious', 'panic', 'overwhelming', 'can\'t cope',
            'breaking down', 'falling apart', 'spiraling'
        ]
        
        text_lower = text.lower()
        
        # Check for high-risk indicators
        if any(keyword in text_lower for keyword in high_risk_keywords):
            return 'high'
        
        # Check emotions and sentiment
        if emotions:
            top_emotion = emotions[0]
            if top_emotion['emotion'] in ['sad', 'anxious'] and top_emotion['confidence'] > 0.7:
                if any(keyword in text_lower for keyword in medium_risk_keywords):
                    return 'high'
                return 'medium'
        
        # Check sentiment
        if sentiment_score < 0.3:  # Very negative
            return 'medium'
        
        return 'low'
    
    def generate_insights(self, text, sentiment, emotions, risk_level):
        """
        Generate human-readable insights
        """
        insights = []
        
        # Sentiment insight
        if sentiment['sentiment'] == 'positive':
            insights.append(f"Your message reflects a positive outlook (confidence: {sentiment['confidence']:.0%})")
        elif sentiment['sentiment'] == 'negative':
            insights.append(f"Your message shows some challenging emotions (confidence: {sentiment['confidence']:.0%})")
        
        # Emotion insights
        if emotions:
            top_emotion = emotions[0]
            insights.append(f"Primary emotion detected: {top_emotion['emotion'].title()} ({top_emotion['confidence']:.0%} confidence)")
            
            if len(emotions) > 1:
                secondary = emotions[1]
                insights.append(f"Also sensing: {secondary['emotion'].title()}")
        
        # Risk-based recommendations
        if risk_level == 'high':
            insights.append("⚠️ Your message suggests significant distress. Please consider reaching out to a mental health professional or crisis helpline.")
        elif risk_level == 'medium':
            insights.append("It seems you're going through a tough time. Remember, it's okay to ask for support.")
        
        return insights

# Initialize the analyzer
analyzer = MentalHealthBERTAnalyzer()

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MindEase AI Service',
        'model': 'BERT-based',
        'timestamp': datetime.now().isoformat(),
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Main endpoint for comprehensive text analysis
    Expected input: { "text": "your text here" }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if len(text) < 3:
            return jsonify({'error': 'Text too short for analysis'}), 400
        
        if len(text) > 1000:
            text = text[:1000]  # Truncate long texts
        
        # Perform analysis
        logger.info(f"Analyzing text: {text[:50]}...")
        
        sentiment = analyzer.analyze_sentiment(text)
        emotions = analyzer.detect_emotions(text)
        risk_level = analyzer.assess_risk_level(
            text, 
            emotions, 
            sentiment['confidence'] if sentiment['sentiment'] == 'negative' else 1 - sentiment['confidence']
        )
        insights = analyzer.generate_insights(text, sentiment, emotions, risk_level)
        
        response = {
            'success': True,
            'sentiment': sentiment['sentiment'],
            'confidence': sentiment['confidence'],
            'detectedEmotions': [e['emotion'] for e in emotions],
            'emotionScores': emotions,
            'riskLevel': risk_level,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Analysis complete: {sentiment['sentiment']} ({risk_level} risk)")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/sentiment', methods=['POST'])
def sentiment_only():
    """
    Quick sentiment analysis endpoint
    Expected input: { "text": "your text here" }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = analyzer.analyze_sentiment(text)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Sentiment error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/emotions', methods=['POST'])
def emotions_only():
    """
    Emotion detection endpoint
    Expected input: { "text": "your text here" }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        emotions = analyzer.detect_emotions(text)
        return jsonify({'emotions': emotions})
    
    except Exception as e:
        logger.error(f"Emotion detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple texts at once
    Expected input: { "texts": ["text1", "text2", ...] }
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid input. Expected array of texts'}), 400
        
        results = []
        for text in texts[:50]:  # Limit to 50 texts
            if text and len(text.strip()) >= 3:
                sentiment = analyzer.analyze_sentiment(text)
                emotions = analyzer.detect_emotions(text)
                risk_level = analyzer.assess_risk_level(text, emotions, sentiment['confidence'])
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment['sentiment'],
                    'confidence': sentiment['confidence'],
                    'primaryEmotion': emotions[0]['emotion'] if emotions else 'neutral',
                    'riskLevel': risk_level
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================
# PLACEHOLDER FOR YOUR PSYCHOLOGICAL MODEL
# ============================================

@app.route('/psychological-assessment', methods=['POST'])
def psychological_assessment():
    """
    Endpoint for your psychological model integration (tomorrow)
    This is where you'll add your specialized mental health assessment model
    """
    return jsonify({
        'message': 'Psychological model endpoint - to be implemented',
        'status': 'pending',
        'note': 'Add your custom mental health assessment model here tomorrow'
    })

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 MindEase AI Service - BERT-based Analysis")
    print("=" * 60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"Server starting on http://localhost:8000")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )