import re
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urlparse
import tldextract
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPhishingDetector:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the phishing detector with improved architecture
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ml_model = pipeline("text-classification", model=model_name)
        self.max_tokens = 512
        
        # Enhanced suspicious keywords with categories
        self.suspicious_keywords = {
            'urgency': ['urgent', 'immediate', 'act now', 'expires today', 'limited time', 'hurry'],
            'security': ['verify', 'security alert', 'unusual activity', 'suspended', 'locked', 'compromised'],
            'credentials': ['password', 'login', 'username', 'confirm identity', 'update account'],
            'financial': ['bank', 'invoice', 'payment', 'refund', 'tax', 'prize', 'winner', 'claim'],
            'actions': ['click here', 'download', 'install', 'update now', 'verify now'],
            'threats': ['suspend', 'terminate', 'close account', 'legal action', 'penalty']
        }
        
        # Legitimate domain patterns (this should ideally be a comprehensive whitelist)
        self.legitimate_domains = {
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
            'paypal.com', 'ebay.com', 'linkedin.com', 'twitter.com', 'instagram.com'
        }
        
        # Phishing domain patterns
        self.phishing_patterns = [
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
            r'[a-zA-Z0-9\-]+\.tk$',  # Free TLD
            r'[a-zA-Z0-9\-]+\.ml$',  # Free TLD
            r'bit\.ly|tinyurl\.com|t\.co',  # URL shorteners
        ]

    def extract_advanced_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive features for phishing detection
        """
        features = {}
        
        # Text-based features
        features.update(self._text_features(text))
        
        # URL-based features
        features.update(self._url_features(text))
        
        # Linguistic features
        features.update(self._linguistic_features(text))
        
        # Structural features
        features.update(self._structural_features(text))
        
        return features

    def _text_features(self, text: str) -> Dict[str, float]:
        """Extract text-based features"""
        text_lower = text.lower()
        
        # Keyword scoring by category
        keyword_scores = {}
        for category, keywords in self.suspicious_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_scores[f'keywords_{category}'] = score
        
        # Overall keyword density
        total_keywords = sum(keyword_scores.values())
        word_count = len(text.split())
        keyword_scores['keyword_density'] = total_keywords / max(word_count, 1)
        
        return keyword_scores

    def _url_features(self, text: str) -> Dict[str, float]:
        """Extract URL-based features"""
        features = {}
        
        # Find all URLs
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        urls = url_pattern.findall(text)
        
        features['url_count'] = len(urls)
        features['has_suspicious_url'] = 0
        features['has_ip_url'] = 0
        features['has_shortened_url'] = 0
        features['suspicious_tld'] = 0
        
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                
                # Check for IP addresses
                if re.match(r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', domain):
                    features['has_ip_url'] = 1
                
                # Check for suspicious patterns
                for pattern in self.phishing_patterns:
                    if re.search(pattern, url, re.IGNORECASE):
                        features['has_suspicious_url'] = 1
                        break
                
                # Check TLD
                extracted = tldextract.extract(url)
                if extracted.suffix in ['tk', 'ml', 'ga', 'cf']:
                    features['suspicious_tld'] = 1
                
                # Check for URL shorteners
                if any(shortener in domain for shortener in ['bit.ly', 'tinyurl.com', 't.co']):
                    features['has_shortened_url'] = 1
                    
            except Exception as e:
                logger.warning(f"Error parsing URL {url}: {e}")
                continue
        
        return features

    def _linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features"""
        features = {}
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['excessive_punctuation'] = int(
            text.count('!') > 3 or text.count('?') > 3 or '!!!' in text
        )
        
        # Capitalization patterns
        words = text.split()
        if words:
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            features['caps_ratio'] = caps_words / len(words)
            features['excessive_caps'] = int(caps_words / len(words) > 0.3)
        else:
            features['caps_ratio'] = 0
            features['excessive_caps'] = 0
        
        # Grammar heuristics (simple)
        features['multiple_dots'] = text.count('...')
        features['multiple_questions'] = text.count('???')
        features['spelling_errors'] = int('???' in text or '...' in text)
        
        return features

    def _structural_features(self, text: str) -> Dict[str, float]:
        """Extract structural features"""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0
            
        # HTML/Email specific
        features['has_html'] = int(bool(re.search(r'<[^>]+>', text)))
        features['has_attachments'] = int('attachment' in text.lower())
        
        return features

    def calculate_rule_based_score(self, features: Dict[str, float]) -> float:
        """
        Calculate rule-based score using weighted features
        """
        # Define weights for different feature categories
        weights = {
            'keywords_urgency': 2.0,
            'keywords_security': 1.5,
            'keywords_credentials': 1.5,
            'keywords_financial': 1.2,
            'keywords_actions': 1.0,
            'keywords_threats': 2.0,
            'has_suspicious_url': 3.0,
            'has_ip_url': 2.5,
            'has_shortened_url': 1.5,
            'suspicious_tld': 2.0,
            'excessive_punctuation': 1.0,
            'excessive_caps': 1.0,
            'keyword_density': 5.0
        }
        
        score = 0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
        
        return score

    def get_ml_prediction(self, text: str) -> Tuple[str, float]:
        """
        Get ML model prediction with proper text preprocessing, handling long texts by chunking.
        """
        tokens = self.tokenizer.encode(text)
        chunk_size = self.max_tokens
        scores = []
        labels = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            # Decode chunk to text
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            # Re-tokenize and truncate to max_tokens to guarantee model safety
            inputs = self.tokenizer(
                chunk_text,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            )
            safe_chunk_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            ml_result = self.ml_model(safe_chunk_text)[0]
            labels.append(ml_result['label'])
            scores.append(ml_result['score'])
        # Aggregate: take the label with the highest average score
        if scores:
            avg_score = sum(scores) / len(scores)
            label = max(set(labels), key=labels.count)
            return label, avg_score
        else:
            return "NEUTRAL", 0.0
    def predict(self, text: str, rule_threshold: float = 4.0, ml_threshold: float = 0.98) -> Dict:
        """
        Main prediction function with ensemble approach
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Extract features
        features = self.extract_advanced_features(text)
        
        # Rule-based prediction
        rule_score = self.calculate_rule_based_score(features)
        rule_prediction = rule_score >= rule_threshold
        
        # ML-based prediction
        ml_label, ml_confidence = self.get_ml_prediction(text)
        
        # Note: The current model is sentiment analysis, not phishing detection
        # In production, you'd use a model specifically trained on phishing data
        ml_prediction = ml_label == "NEGATIVE" and ml_confidence > ml_threshold
        
        # Ensemble decision
        final_prediction = rule_prediction or ml_prediction
        
        # Calculate confidence score
        confidence = min((rule_score / 10.0) + (ml_confidence * 0.5), 1.0)
        
        return {
            'is_phishing': final_prediction,
            'confidence': confidence,
            'rule_score': rule_score,
            'ml_label': ml_label,
            'ml_confidence': ml_confidence,
            'features': features
        }

    def explain_prediction(self, text: str) -> str:
        """
        Provide explanation for the prediction
        """
        result = self.predict(text)
        
        explanation = f"Prediction: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}\n"
        explanation += f"Confidence: {result['confidence']:.2f}\n"
        explanation += f"Rule-based score: {result['rule_score']:.2f}\n\n"
        
        # Highlight key features
        features = result['features']
        if features.get('keywords_urgency', 0) > 0:
            explanation += "- Contains urgency keywords\n"
        if features.get('has_suspicious_url', 0):
            explanation += "- Contains suspicious URLs\n"
        if features.get('excessive_punctuation', 0):
            explanation += "- Excessive punctuation detected\n"
        if features.get('keyword_density', 0) > 0.1:
            explanation += f"- High suspicious keyword density: {features['keyword_density']:.2f}\n"
            
        return explanation


def main():
    """Example usage and testing"""
    detector = ImprovedPhishingDetector()
    
    test_texts = [
        "URGENT: Your account has been suspended! Click here to verify your information at http://192.168.1.1/login",
        "Hey, are we still on for lunch tomorrow?",
        "Security alert: Unusual activity detected. Please login immediately to update your password.",
        "Congratulations! You've won $1,000,000! Click here to claim your prize NOW!!!",
        "Your Amazon order #123456 has been shipped and will arrive tomorrow.",
        "IMMEDIATE ACTION REQUIRED: Your PayPal account will be closed. Verify now: bit.ly/verify123"
    ]
    
    print("Phishing Detection Results:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print("-" * 30)
        
        result = detector.predict(text)
        print(f"Prediction: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Rule Score: {result['rule_score']:.2f}")
        print(f"ML Prediction: {result['ml_label']} ({result['ml_confidence']:.2f})")


if __name__ == "__main__":
    main()