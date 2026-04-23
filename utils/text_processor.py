"""
Text processing utilities for ESG analysis
"""

import logging
import re
import ssl
from typing import List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

# Required NLTK packages and their resource paths for verification
_NLTK_REQUIRED: dict = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
}


class TextProcessor:
    """Text preprocessing and cleaning utilities."""

    def __init__(self) -> None:
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words: Set[str] = set(stopwords.words('english'))

    @staticmethod
    def _download_nltk_data() -> None:
        """Download required NLTK data packages with SSL workaround."""
        # Fix SSL certificate issues in some environments
        try:
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
        except AttributeError:
            pass

        for package, resource_path in _NLTK_REQUIRED.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info("Downloading NLTK package: %s", package)
                nltk.download(package, quiet=True)

        logger.info("NLTK data ready")

    def clean_text(self, text: Optional[str]) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.

        Args:
            text: Input text.

        Returns:
            Cleaned text, or empty string if input is None/empty.
        """
        if not text:
            return ""
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r"[^\w\s.,;:\-'\"()]", ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_sentences(self, text: Optional[str], min_length: int = 100) -> List[str]:
        """
        Extract sentences from text with a minimum length filter.

        Args:
            text: Input text.
            min_length: Minimum character length for a sentence to be included.

        Returns:
            List of sentences meeting the length requirement.
        """
        if not text:
            return []

        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > min_length]
        except Exception:
            logger.debug("sent_tokenize failed, using regex fallback", exc_info=True)
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > min_length]

    def lemmatize_text(self, text: Optional[str]) -> str:
        """
        Lemmatize and stem text for topic modeling.

        Args:
            text: Input text.

        Returns:
            Space-separated string of processed tokens.
        """
        if not text:
            return ""

        try:
            tokens = word_tokenize(text.lower())
            processed: List[str] = []
            for token in tokens:
                if token.isalpha() and len(token) > 3 and token not in self.stop_words:
                    lemma = self.lemmatizer.lemmatize(token)
                    stem = self.stemmer.stem(lemma)
                    processed.append(stem)
            return ' '.join(processed)
        except Exception:
            logger.debug("lemmatize_text failed", exc_info=True)
            return ""

    @staticmethod
    def match_keywords_regex(
        sentence: Optional[str], keywords: List[str]
    ) -> Optional[str]:
        """
        Match keywords using strict word-boundary regex.

        Args:
            sentence: Input sentence.
            keywords: List of keywords to match.

        Returns:
            The first matched keyword, or None.
        """
        if not sentence:
            return None

        sentence_lower = sentence.lower()
        for keyword in keywords:
            pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(pattern, sentence_lower):
                return keyword
        return None

    def add_custom_stopwords(self, additional: List[str]) -> None:
        """Add custom stopwords to the internal set."""
        self.stop_words.update(additional)

    def get_stopwords(self) -> Set[str]:
        """Return the current set of stopwords."""
        return self.stop_words