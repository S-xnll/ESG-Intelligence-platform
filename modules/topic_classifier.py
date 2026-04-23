"""
Module 4: ESG Topic Classification
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import ESG_TOPICS, OUTPUT_DIR
from utils import DataLoader

logger = logging.getLogger(__name__)


class TopicClassifier:
    """Supervised ESG Topic Classification with optional zero-shot fallback."""

    def __init__(
        self,
        statements_df: Optional[pd.DataFrame] = None,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.statements_df = statements_df
        self.output_dir = Path(output_dir)
        self.data_loader = DataLoader()

        self.esg_topics = ESG_TOPICS
        self.classifier: Optional[Pipeline] = None
        self.predictions: Optional[pd.DataFrame] = None
        self.zero_shot_classifier: Any = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data loading ----------

    def load_statements(self, statements_file: Optional[Path] = None) -> None:
        """Load statements from CSV. Falls back to default location."""
        if statements_file:
            self.statements_df = self.data_loader.load_csv(statements_file)
        elif self.statements_df is None:
            default_path = self.output_dir / 'statements.csv'
            if default_path.exists():
                self.statements_df = self.data_loader.load_csv(default_path)
            else:
                raise FileNotFoundError(
                    f"No statements data available. Expected at {default_path}"
                )
        logger.info("Loaded %d statements", len(self.statements_df))

    # ---------- training ----------

    def create_training_data(
        self, sample_size: int = 500
    ) -> Tuple[List[str], List[str]]:
        """
        Create pseudo-labeled training data via keyword matching.

        Args:
            sample_size: Number of statements to sample.

        Returns:
            Tuple of (texts, labels).
        """
        logger.info("Creating training data from up to %d samples...", sample_size)

        if self.statements_df is None:
            self.load_statements()

        sample_df = (
            self.statements_df.sample(n=sample_size, random_state=42)
            if len(self.statements_df) > sample_size
            else self.statements_df
        )

        texts: List[str] = []
        labels: List[str] = []

        for _, row in sample_df.iterrows():
            text_lower = row['statement'].lower()
            for topic, keywords in self.esg_topics.items():
                if any(kw in text_lower for kw in keywords):
                    texts.append(row['statement'])
                    labels.append(topic)
                    break  # first match wins

        logger.info("Labeled %d statements across %d unique topics",
                     len(texts), len(set(labels)))
        return texts, labels

    def train_supervised_classifier(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Optional[Pipeline]:
        """Train an SVM classifier. Returns None if insufficient data."""
        if texts is None or labels is None:
            texts, labels = self.create_training_data()

        if len(texts) < 10:
            logger.warning("Insufficient training data (%d samples)", len(texts))
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        self.classifier = Pipeline([
            (
                'tfidf',
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english',
                ),
            ),
            ('clf', LinearSVC(random_state=42, max_iter=2000)),
        ])
        self.classifier.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, self.classifier.predict(X_train))
        test_acc = accuracy_score(y_test, self.classifier.predict(X_test))
        logger.info("Training accuracy: %.2f%%, Test accuracy: %.2f%%",
                     train_acc * 100, test_acc * 100)

        return self.classifier

    # ---------- classification ----------

    def classify_statements(
        self,
        method: str = 'auto',
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Classify statements using the chosen method.

        Args:
            method: 'supervised', 'keyword', or 'auto'
                    (tries supervised first).
            sample_size: Limit classification to a random sample.

        Returns:
            DataFrame with classification results.
        """
        if self.statements_df is None:
            self.load_statements()

        classify_df = (
            self.statements_df.sample(n=sample_size, random_state=42)
            if sample_size
            else self.statements_df
        )

        if method == 'auto':
            if self.classifier is None:
                self.train_supervised_classifier()
            method = 'supervised' if self.classifier is not None else 'keyword'

        if method == 'supervised' and self.classifier is not None:
            results = self._classify_supervised(classify_df)
        else:
            results = self._classify_keyword(classify_df)

        self.predictions = pd.DataFrame(results)
        self._log_summary()
        return self.predictions

    def get_company_topic_profile(self) -> pd.DataFrame:
        """Return topic distribution (percentages) per company."""
        if self.predictions is None:
            self.classify_statements()

        profile = (
            self.predictions
            .groupby(['stock_code', 'predicted_topic'])
            .size()
            .reset_index(name='count')
        )
        pivot = profile.pivot(
            index='stock_code', columns='predicted_topic', values='count'
        ).fillna(0)
        return pivot.div(pivot.sum(axis=1), axis=0) * 100

    def save_results(self) -> None:
        """Persist classification results to CSV."""
        if self.predictions is not None:
            self.data_loader.save_csv(
                self.predictions,
                self.output_dir / 'topic_classifications.csv',
            )
            profile = self.get_company_topic_profile()
            self.data_loader.save_csv(
                profile.reset_index(),
                self.output_dir / 'company_topic_profile.csv',
            )
        logger.info("Classification results saved")

    def run_full_classification(
        self,
        method: str = 'auto',
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Convenience method: load, classify, save."""
        logger.info("=" * 60)
        logger.info("TOPIC CLASSIFICATION MODULE")
        logger.info("=" * 60)

        if self.statements_df is None:
            self.load_statements()

        results = self.classify_statements(method=method, sample_size=sample_size)
        self.save_results()
        return results

    # ---------- internal ----------

    def _classify_supervised(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Classify using the trained SVM pipeline."""
        texts = df['statement'].tolist()
        predictions = self.classifier.predict(texts)
        return [
            {
                'stock_code': row.stock_code,
                'statement': row.statement[:300],
                'predicted_topic': pred,
                'method': 'supervised',
            }
            for pred, (_, row) in zip(predictions, df.iterrows())
        ]

    def _classify_keyword(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Classify using simple keyword matching."""
        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            text_lower = row['statement'].lower()
            topic_scores: Dict[str, int] = {}
            for topic, keywords in self.esg_topics.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    topic_scores[topic] = score

            if topic_scores:
                predicted = max(topic_scores, key=topic_scores.get)
                confidence = topic_scores[predicted] / sum(topic_scores.values())
            else:
                predicted = 'Unclassified'
                confidence = 0.0

            results.append({
                'stock_code': row['stock_code'],
                'statement': row['statement'][:300],
                'predicted_topic': predicted,
                'confidence': confidence,
                'method': 'keyword',
            })
        return results

    def _log_summary(self) -> None:
        """Log a summary of classification results."""
        if self.predictions is None:
            return
        logger.info("Classified %d statements", len(self.predictions))
        if 'predicted_topic' in self.predictions.columns:
            counts = self.predictions['predicted_topic'].value_counts()
            logger.info("Topic distribution:\n%s", counts.head(10).to_string())


def main() -> pd.DataFrame:
    """Entry point for standalone execution."""
    classifier = TopicClassifier()
    return classifier.run_full_classification()


if __name__ == "__main__":
    main()