"""
ESG Analysis Modules
"""

from .pdf_extractor import PDFExtractor
from .topic_classifier import TopicClassifier

# Optional modules -- import only if they exist
try:
    from .sentiment_analyzer import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None

try:
    from .advanced_analytics import AdvancedAnalytics
except ImportError:
    AdvancedAnalytics = None

try:
    from .esg_predictor import ESGPredictor
except ImportError:
    ESGPredictor = None

try:
    from .chatbot_integration import ESGChatbot
except ImportError:
    ESGChatbot = None

__all__ = [
    'PDFExtractor',
    'TopicClassifier',
    'SentimentAnalyzer',
    'AdvancedAnalytics',
    'ESGPredictor',
    'ESGChatbot',
]