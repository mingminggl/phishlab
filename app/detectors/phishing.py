import re
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import urlparse
import tldextract
from typing import Dict, List, Tuple, Optional
import logging
import warnings
from statistics import mean
import hashlib

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ImprovedPhishingDetector:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the phishing detector with improved architecture
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ml_model = pipeline("text-classification", model=model_name, return_all_scores=True)
        self.max_tokens = 512
        
        # Enhanced suspicious keywords with categories and weights
        self.suspicious_keywords = {
            'urgency': {
                'keywords': ['urgent', 'immediate', 'act now', 'expires today', 'limited time', 'hurry', 'asap', 'deadline', 'expire soon', 'time sensitive'],
                'weight': 2.5
            },
            'security': {
                'keywords': ['verify', 'security alert', 'unusual activity', 'suspended', 'locked', 'compromised', 'breach', 'unauthorized', 'violation', 'restricted'],
                'weight': 2.0
            },
            'credentials': {
                'keywords': ['password', 'login', 'username', 'confirm identity', 'update account', 'credentials', 'authentication', 'pin', 'security code'],
                'weight': 2.0
            },
            'financial': {
                'keywords': ['bank', 'invoice', 'payment', 'refund', 'tax', 'prize', 'winner', 'claim', 'reward', 'money', 'cash', 'transfer', 'account', 'credit card'],
                'weight': 1.8
            },
            'actions': {
                'keywords': ['click here', 'download', 'install', 'update now', 'verify now', 'confirm now', 'submit', 'proceed', 'continue', 'activate'],
                'weight': 1.5
            },
            'threats': {
                'keywords': ['suspend', 'terminate', 'close account', 'legal action', 'penalty', 'consequences', 'blocked', 'disabled', 'frozen'],
                'weight': 2.5
            },
            'social_engineering': {
                'keywords': ['congratulations', 'selected', 'chosen', 'lucky', 'exclusive', 'special offer', 'limited offer', 'deal', 'discount'],
                'weight': 1.5
            }
        }
        
        # Expanded legitimate domain patterns
        self.legitimate_domains = {
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
            'paypal.com', 'ebay.com', 'linkedin.com', 'twitter.com', 'instagram.com',
            'github.com', 'stackoverflow.com', 'reddit.com', 'youtube.com', 'gmail.com',
            'outlook.com', 'yahoo.com', 'dropbox.com', 'slack.com', 'zoom.us'
        }
        
        # Enhanced phishing domain patterns
        self.phishing_patterns = [
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
            r'[a-zA-Z0-9\-]+\.(tk|ml|ga|cf|gq)$',  # Free TLDs
            r'bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|short\.link',  # URL shorteners
            r'[a-zA-Z0-9\-]*paypal[a-zA-Z0-9\-]*\.(com|net|org)(?!paypal\.com)',  # Paypal typosquatting
            r'[a-zA-Z0-9\-]*amazon[a-zA-Z0-9\-]*\.(com|net|org)(?!amazon\.com)',  # Amazon typosquatting
            r'[a-zA-Z0-9\-]*microsoft[a-zA-Z0-9\-]*\.(com|net|org)(?!microsoft\.com)',  # Microsoft typosquatting
        ]
        
        # Suspicious TLDs
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'ws', 'info', 'biz'
        }

    def safe_url_parse(self, url: str) -> Optional[Tuple[str, str, str]]:
        """
        Safely parse URL and return (scheme, netloc, path) or None if invalid
        """
        try:
            # Clean the URL first
            url = url.strip()
            if not url:
                return None
                
            # Handle malformed URLs
            if url.startswith('http') and '[' in url:
                url = url.split('[')[0]
            if url.startswith('http') and ']' in url:
                url = url.split(']')[0]
                
            # Add scheme if missing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
                
            parsed = urlparse(url)
            if parsed.netloc:
                return (parsed.scheme, parsed.netloc.lower(), parsed.path)
            return None
        except Exception:
            return None

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
        
        # Advanced pattern features
        features.update(self._pattern_features(text))
        
        return features

    def _text_features(self, text: str) -> Dict[str, float]:
        """Extract text-based features with improved keyword matching"""
        text_lower = text.lower()
        
        # Keyword scoring by category with weights
        keyword_scores = {}
        total_weighted_score = 0
        
        for category, data in self.suspicious_keywords.items():
            keywords = data['keywords']
            weight = data['weight']
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_scores[f'keywords_{category}'] = matches
            total_weighted_score += matches * weight
        
        # Overall keyword metrics
        word_count = len(text.split())
        total_keywords = sum(keyword_scores.values())
        
        keyword_scores['keyword_density'] = total_keywords / max(word_count, 1)
        keyword_scores['weighted_keyword_score'] = total_weighted_score
        keyword_scores['keyword_diversity'] = len([k for k in keyword_scores.values() if k > 0])
        
        return keyword_scores

    def _url_features(self, text: str) -> Dict[str, float]:
        """Extract URL-based features with improved parsing"""
        features = {}
        
        # Enhanced URL pattern
        url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+|[a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,}[^\s<>"{}|\\^`\[\]]*',
            re.IGNORECASE
        )
        
        potential_urls = url_pattern.findall(text)
        
        # Initialize features
        features.update({
            'url_count': 0,
            'has_suspicious_url': 0,
            'has_ip_url': 0,
            'has_shortened_url': 0,
            'suspicious_tld': 0,
            'typosquatting_score': 0,
            'url_length_avg': 0,
            'suspicious_url_patterns': 0
        })
        
        valid_urls = []
        url_lengths = []
        
        for url in potential_urls:
            parsed_info = self.safe_url_parse(url)
            if parsed_info is None:
                continue
                
            scheme, domain, path = parsed_info
            valid_urls.append((scheme, domain, path))
            url_lengths.append(len(url))
            
            # Check for IP addresses
            if re.match(r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', domain):
                features['has_ip_url'] = 1
            
            # Check for suspicious patterns
            for pattern in self.phishing_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    features['has_suspicious_url'] = 1
                    features['suspicious_url_patterns'] += 1
                    break
            
            # Check TLD
            try:
                extracted = tldextract.extract(url)
                if extracted.suffix.lower() in self.suspicious_tlds:
                    features['suspicious_tld'] = 1
            except:
                pass
            
            # Check for URL shorteners
            shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
            if any(shortener in domain for shortener in shorteners):
                features['has_shortened_url'] = 1
            
            # Typosquatting detection
            features['typosquatting_score'] += self._check_typosquatting(domain)
        
        features['url_count'] = len(valid_urls)
        features['url_length_avg'] = mean(url_lengths) if url_lengths else 0
        
        return features

    def _check_typosquatting(self, domain: str) -> float:
        """Check for typosquatting against legitimate domains"""
        score = 0
        domain_lower = domain.lower()
        
        for legit_domain in self.legitimate_domains:
            # Check if domain contains legitimate domain name
            if legit_domain.replace('.com', '') in domain_lower and domain_lower != legit_domain:
                score += 1
            
            # Check for character substitution
            legit_name = legit_domain.split('.')[0]
            domain_name = domain_lower.split('.')[0]
            
            if len(legit_name) > 3 and len(domain_name) > 3:
                # Simple edit distance check
                if self._simple_edit_distance(legit_name, domain_name) <= 2 and legit_name != domain_name:
                    score += 0.5
        
        return min(score, 3.0)  # Cap at 3.0

    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance calculation"""
        if len(s1) < len(s2):
            return self._simple_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features"""
        features = {}
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['excessive_punctuation'] = int(
            text.count('!') > 2 or text.count('?') > 2 or '!!!' in text or '???' in text
        )
        
        # Capitalization patterns
        words = text.split()
        if words:
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            features['caps_ratio'] = caps_words / len(words)
            features['excessive_caps'] = int(caps_words / len(words) > 0.2)
            
            # All caps sentences
            sentences = re.split(r'[.!?]+', text)
            caps_sentences = sum(1 for sentence in sentences if sentence.strip().isupper() and len(sentence.strip()) > 10)
            features['caps_sentences'] = caps_sentences
        else:
            features['caps_ratio'] = 0
            features['excessive_caps'] = 0
            features['caps_sentences'] = 0
        
        # Grammar and spelling heuristics
        features['multiple_dots'] = text.count('...')
        features['multiple_questions'] = text.count('???')
        features['spelling_errors'] = int('???' in text or '...' in text)
        
        # Repetitive patterns
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
        
        return features

    def _structural_features(self, text: str) -> Dict[str, float]:
        """Extract structural features"""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            features['long_words'] = sum(1 for word in words if len(word) > 8) / len(words)
        else:
            features['avg_word_length'] = 0
            features['long_words'] = 0
            
        # HTML/Email specific
        features['has_html'] = int(bool(re.search(r'<[^>]+>', text)))
        features['has_attachments'] = int('attachment' in text.lower())
        
        # Readability approximation
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
        
        return features

    def _pattern_features(self, text: str) -> Dict[str, float]:
        """Extract advanced pattern features"""
        features = {}
        
        # Phone number patterns
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b'
        features['phone_numbers'] = len(re.findall(phone_pattern, text))
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        features['email_addresses'] = len(re.findall(email_pattern, text))
        
        # Cryptocurrency patterns
        crypto_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,59}\b'
        features['crypto_addresses'] = len(re.findall(crypto_pattern, text))
        
        # Suspicious file extensions
        malicious_extensions = ['exe', 'scr', 'bat', 'com', 'pif', 'vbs', 'js']
        features['malicious_extensions'] = sum(1 for ext in malicious_extensions if f'.{ext}' in text.lower())
        
        return features

    def calculate_rule_based_score(self, features: Dict[str, float]) -> float:
        """
        Calculate rule-based score using optimized weighted features
        """
        # Optimized weights based on phishing detection research
        weights = {
            # Keyword categories
            'weighted_keyword_score': 0.8,
            'keyword_diversity': 1.0,
            'keyword_density': 3.0,
            
            # URL features
            'has_suspicious_url': 4.0,
            'has_ip_url': 3.5,
            'has_shortened_url': 2.0,
            'suspicious_tld': 2.5,
            'typosquatting_score': 2.0,
            'suspicious_url_patterns': 1.5,
            
            # Linguistic features
            'excessive_punctuation': 1.5,
            'excessive_caps': 1.8,
            'caps_sentences': 2.0,
            'repeated_chars': 1.0,
            
            # Structural features
            'malicious_extensions': 3.0,
            'crypto_addresses': 2.5,
            
            # Pattern features
            'phone_numbers': 0.5,
            'email_addresses': 0.3,
        }
        
        score = 0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
        
        return score

    def get_ml_prediction(self, text: str) -> Tuple[str, float]:
        """
        Get ML model prediction with robust text chunking
        """
        try:
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                return "NEUTRAL", 0.5
            
            # Tokenize and check length
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= self.max_tokens - 2:  # Account for special tokens
                # Text fits in one chunk
                result = self.ml_model(text)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Find the score for negative sentiment (proxy for phishing)
                        negative_score = next((item['score'] for item in result[0] if item['label'] == 'NEGATIVE'), 0.5)
                        return 'NEGATIVE' if negative_score > 0.5 else 'POSITIVE', negative_score
                    else:
                        return result[0]['label'], result[0]['score']
            else:
                # Split into chunks
                chunk_size = self.max_tokens - 2
                chunks = []
                
                for i in range(0, len(tokens), chunk_size):
                    chunk_tokens = tokens[i:i + chunk_size]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk_text)
                
                # Process each chunk
                negative_scores = []
                for chunk in chunks:
                    if chunk.strip():
                        try:
                            result = self.ml_model(chunk)
                            if isinstance(result, list) and len(result) > 0:
                                if isinstance(result[0], list):
                                    negative_score = next((item['score'] for item in result[0] if item['label'] == 'NEGATIVE'), 0.5)
                                    negative_scores.append(negative_score)
                                else:
                                    score = result[0]['score'] if result[0]['label'] == 'NEGATIVE' else 1 - result[0]['score']
                                    negative_scores.append(score)
                        except Exception:
                            continue
                
                # Aggregate scores
                if negative_scores:
                    avg_negative = mean(negative_scores)
                    return 'NEGATIVE' if avg_negative > 0.5 else 'POSITIVE', avg_negative
                
            return "NEUTRAL", 0.5
            
        except Exception as e:
            logger.warning(f"Error in ML prediction: {e}")
            return "NEUTRAL", 0.5

    def predict(self, text: str, rule_threshold: float = 8.0, ml_threshold: float = 0.98) -> Dict:
        """
        Main prediction function with optimized ensemble approach
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
        ml_prediction = ml_label == "NEGATIVE" and ml_confidence > ml_threshold
        
        # Enhanced ensemble decision
        # Give more weight to rule-based system for phishing detection
        confidence_score = 0.0
        
        if rule_prediction and ml_prediction:
            final_prediction = True
            confidence_score = min(0.9, (rule_score / 15.0) + (ml_confidence * 0.3))
        elif rule_prediction:
            final_prediction = True
            confidence_score = min(0.8, rule_score / 15.0)
        elif ml_prediction:
            final_prediction = True
            confidence_score = min(0.7, ml_confidence * 0.8)
        else:
            final_prediction = False
            confidence_score = max(0.1, 1.0 - ((rule_score / 15.0) + (ml_confidence * 0.3)))
        
        return {
            'is_phishing': final_prediction,
            'confidence': confidence_score,
            'rule_score': rule_score,
            'ml_label': ml_label,
            'ml_confidence': ml_confidence,
            'features': features
        }

    def explain_prediction(self, text: str) -> str:
        """
        Provide detailed explanation for the prediction
        """
        result = self.predict(text)
        
        explanation = f"Prediction: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}\n"
        explanation += f"Confidence: {result['confidence']:.3f}\n"
        explanation += f"Rule-based score: {result['rule_score']:.2f}\n"
        explanation += f"ML prediction: {result['ml_label']} ({result['ml_confidence']:.3f})\n\n"
        
        # Highlight key features
        features = result['features']
        explanations = []
        
        if features.get('weighted_keyword_score', 0) > 5:
            explanations.append(f"High weighted keyword score: {features['weighted_keyword_score']:.1f}")
        
        if features.get('has_suspicious_url', 0):
            explanations.append("Contains suspicious URLs")
        
        if features.get('has_ip_url', 0):
            explanations.append("Contains IP-based URLs")
        
        if features.get('typosquatting_score', 0) > 0:
            explanations.append(f"Potential typosquatting detected: {features['typosquatting_score']:.1f}")
        
        if features.get('excessive_punctuation', 0):
            explanations.append("Excessive punctuation detected")
        
        if features.get('excessive_caps', 0):
            explanations.append("Excessive capitalization detected")
        
        if features.get('keyword_density', 0) > 0.15:
            explanations.append(f"High suspicious keyword density: {features['keyword_density']:.3f}")
        
        if explanations:
            explanation += "Key indicators:\n"
            for exp in explanations:
                explanation += f"- {exp}\n"
        
        return explanation


def main():
    """Example usage and testing"""
    detector = ImprovedPhishingDetector()
    
    test_texts = [
        "URGENT: Your account has been suspended! Click here to verify your information at http://192.168.1.1/login",
        "Hey, are we still on for lunch tomorrow?",
        "Security alert: Unusual activity detected. Please login immediately to update your password.",
        "Congratulations! You've won $1,000,000! Click here to claim your prize NOW!!! bit.ly/claim123",
        "Your Amazon order #123456 has been shipped and will arrive tomorrow.",
        "IMMEDIATE ACTION REQUIRED: Your PayPal account will be closed. Verify now: paypal-security.tk/verify",
        "Your Microsoft account has been compromised. Download the security patch: microsoft-update.exe"
    ]
    
    print("Enhanced Phishing Detection Results:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print("-" * 40)
        
        result = detector.predict(text)
        print(f"Prediction: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Rule Score: {result['rule_score']:.2f}")
        print(f"ML: {result['ml_label']} ({result['ml_confidence']:.3f})")
        
        # Show top features
        features = result['features']
        important_features = [(k, v) for k, v in features.items() if v > 0 and k in [
            'weighted_keyword_score', 'has_suspicious_url', 'typosquatting_score', 
            'keyword_density', 'excessive_caps', 'suspicious_url_patterns'
        ]]
        
        if important_features:
            print("Key features:")
            for feature, value in important_features:
                print(f"  {feature}: {value:.2f}")


if __name__ == "__main__":
    main()