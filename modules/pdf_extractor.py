"""
Module 1: PDF Text Extraction and ESG Sentence Filtering
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import ESG_KEYWORDS, HKEX_DATA_FILE, JSON_DIR, OUTPUT_DIR, PDF_FOLDER
from utils import DataLoader, TextProcessor

logger = logging.getLogger(__name__)


class PDFExtractor:
    """PDF text extraction and ESG-related sentence filtering."""

    def __init__(
        self,
        pdf_folder: Path = PDF_FOLDER,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.pdf_folder = Path(pdf_folder)
        self.output_dir = Path(output_dir)
        self.json_dir = JSON_DIR

        self.text_processor = TextProcessor()
        self.data_loader = DataLoader()
        self.hkex_data = self.data_loader.load_hkex_data(HKEX_DATA_FILE)
        self.esg_keywords = ESG_KEYWORDS

    # ---------- public API ----------

    def process_all_pdfs(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Process all PDFs in the configured folder.

        Returns:
            Tuple of (list of ESG data dicts, combined statements DataFrame).
        """
        logger.info("=" * 60)
        logger.info("PDF EXTRACTION MODULE")
        logger.info("=" * 60)

        if not self.pdf_folder.exists():
            logger.error("Folder not found: %s", self.pdf_folder)
            return [], pd.DataFrame()

        available_stocks = self.data_loader.get_available_pdfs(self.pdf_folder)
        logger.info("Found %d PDF files", len(available_stocks))

        all_data: List[Dict[str, Any]] = []
        all_statements: List[Dict[str, str]] = []

        for stock_code in available_stocks:
            esg_data = self.process_single_pdf(stock_code)
            if esg_data is None:
                continue

            all_data.append(esg_data)
            self.data_loader.save_json(
                esg_data, self.json_dir / f"{stock_code}_esg_texts.json"
            )

            # Collect statements for downstream analysis
            for sentence in esg_data.get('all_sentences', []):
                lemma = self.text_processor.lemmatize_text(sentence)
                if len(lemma) > 50:
                    all_statements.append({
                        'stock_code': stock_code,
                        'stock_name': esg_data.get('stock_name', stock_code),
                        'statement': sentence,
                        'lemma': lemma,
                    })

        # Persist combined outputs
        if all_data:
            self.data_loader.save_json(
                all_data, self.output_dir / 'all_esg_texts_combined.json'
            )
            self._save_summary_csv(all_data)

        statements_df = pd.DataFrame(all_statements)
        if not statements_df.empty:
            self.data_loader.save_csv(
                statements_df, self.output_dir / 'statements.csv'
            )

        logger.info("PDF Extraction Complete -- %d companies, %d statements",
                     len(all_data), len(statements_df))
        return all_data, statements_df

    def process_single_pdf(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF by stock code.

        Args:
            stock_code: Stock code identifier.

        Returns:
            ESG data dictionary or None if processing fails.
        """
        pdf_path = self.pdf_folder / f"{stock_code}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            return None

        logger.info("Processing: %s", stock_code)
        return self._process_pdf(pdf_path, stock_code)

    def process_multiple_pdfs(
        self, stock_codes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs sequentially.

        Args:
            stock_codes: List of stock code identifiers.

        Returns:
            List of successfully processed ESG data dicts.
        """
        results: List[Dict[str, Any]] = []
        for stock_code in stock_codes:
            esg_data = self.process_single_pdf(stock_code)
            if esg_data:
                results.append(esg_data)
                self.data_loader.save_json(
                    esg_data, self.json_dir / f"{stock_code}_esg_texts.json"
                )
        return results

    def get_available_stocks(self) -> pd.DataFrame:
        """Get a DataFrame of stocks that have corresponding PDFs."""
        available = self.data_loader.get_available_pdfs(self.pdf_folder)
        enriched: List[Dict[str, str]] = []
        for code in available:
            info = self.data_loader.get_stock_info(code)
            enriched.append(info)
        return pd.DataFrame(enriched)

    # ---------- public helpers ----------

    def extract_esg_sentences(
        self, sentences: List[str]
    ) -> Dict[str, List[str]]:
        """
        Categorise sentences into E / S / G / Mixed based on keyword hits.

        Args:
            sentences: List of sentences.

        Returns:
            Dictionary with keys 'Environmental', 'Social', 'Governance', 'Mixed'.
        """
        esg_sentences: Dict[str, List[str]] = {
            'Environmental': [],
            'Social': [],
            'Governance': [],
            'Mixed': [],
        }

        for sentence in sentences:
            categories_found = []
            for category, keywords in self.esg_keywords.items():
                if self.text_processor.match_keywords_regex(sentence, keywords):
                    categories_found.append(category)

            if len(categories_found) == 1:
                esg_sentences[categories_found[0]].append(sentence)
            elif len(categories_found) > 1:
                esg_sentences['Mixed'].append(sentence)

        return esg_sentences

    # ---------- internal ----------

    def _process_pdf(
        self, pdf_path: Path, stock_code: str
    ) -> Optional[Dict[str, Any]]:
        """Core PDF processing logic."""
        stock_info = self.data_loader.get_stock_info(stock_code)
        stock_name = stock_info.get('stock_name', stock_code)
        logger.debug("  Stock: %s - %s", stock_code, stock_name)

        text = self.data_loader.extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning("  No text extracted from %s", stock_code)
            return None

        cleaned = self.text_processor.clean_text(text)
        sentences = self.text_processor.extract_sentences(cleaned)
        logger.debug("  Extracted %d sentences", len(sentences))

        if not sentences:
            return None

        esg_sentences = self.extract_esg_sentences(sentences)
        esg_count = sum(len(v) for v in esg_sentences.values())

        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'file_name': f"{stock_code}.pdf",
            'extraction_date': pd.Timestamp.now().isoformat(),
            'statistics': {
                'total_sentences': len(sentences),
                'esg_sentences': esg_count,
                'environmental': len(esg_sentences['Environmental']),
                'social': len(esg_sentences['Social']),
                'governance': len(esg_sentences['Governance']),
                'mixed': len(esg_sentences['Mixed']),
            },
            'esg_texts': esg_sentences,
            'all_sentences': sentences,
        }

    def _save_summary_csv(self, all_data: List[Dict[str, Any]]) -> None:
        """Generate and save a summary CSV of extraction statistics."""
        summary: List[Dict[str, Any]] = []
        for data in all_data:
            stats = data['statistics']
            total = stats['total_sentences']
            esg = stats['esg_sentences']
            summary.append({
                'stock_code': data['stock_code'],
                'stock_name': data.get('stock_name', data['stock_code']),
                'total_sentences': total,
                'esg_sentences': esg,
                'environmental': stats['environmental'],
                'social': stats['social'],
                'governance': stats['governance'],
                'mixed': stats['mixed'],
                'esg_percentage': round(esg / total * 100, 1) if total > 0 else 0,
            })
        self.data_loader.save_csv(
            pd.DataFrame(summary), self.output_dir / 'esg_summary.csv'
        )


def main() -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Entry point for standalone execution."""
    extractor = PDFExtractor()
    return extractor.process_all_pdfs()


if __name__ == "__main__":
    main()