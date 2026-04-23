"""
Data loading utilities for ESG analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import PyPDF2
import pdfplumber

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and file handling utilities"""

    def __init__(self) -> None:
        self.hkex_data: Optional[pd.DataFrame] = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Suppress verbose logging from PDF libraries"""
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    def load_hkex_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load HKEX data CSV file.

        Args:
            filepath: Path to HKEX data CSV.

        Returns:
            DataFrame containing HKEX data; empty DataFrame if loading fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("HKEX data file not found: %s", filepath)
            return pd.DataFrame()

        try:
            self.hkex_data = pd.read_csv(filepath)
            logger.info("Loaded HKEX data: %d records", len(self.hkex_data))
            return self.hkex_data
        except Exception as e:
            logger.error("Error loading HKEX data: %s", e)
            return pd.DataFrame()

    def get_stock_info(self, stock_code: Union[str, int]) -> Dict[str, str]:
        """
        Get stock information by stock code.

        Tries multiple formats: exact match, without leading zeros, padded to 5 digits.

        Args:
            stock_code: Stock code identifier.

        Returns:
            Dictionary with stock_code and stock_name at minimum.
        """
        if self.hkex_data is None or self.hkex_data.empty:
            return {'stock_code': str(stock_code), 'stock_name': str(stock_code)}

        stock_code_str = str(stock_code)
        code_variants = [
            stock_code_str,
            stock_code_str.lstrip('0') or '0',
            stock_code_str.zfill(5),
        ]

        for variant in code_variants:
            matches = self.hkex_data[self.hkex_data['stock_code'].astype(str) == variant]
            if not matches.empty:
                row = matches.iloc[0]
                return {
                    'stock_code': str(stock_code),
                    'stock_name': str(row.get('stock_name', stock_code)),
                    'headline': str(row.get('headline', '')),
                    'doc_title': str(row.get('doc_title', '')),
                    'doc_link': str(row.get('doc_link', '')),
                    'release_time': str(row.get('release_time', '')),
                }

        return {'stock_code': str(stock_code), 'stock_name': str(stock_code)}

    def search_stock_by_name(self, name_query: str) -> pd.DataFrame:
        """
        Search stock by name substring (case-insensitive).

        Args:
            name_query: Name query string.

        Returns:
            DataFrame of matching stocks with stock_code and stock_name.
        """
        if self.hkex_data is None or self.hkex_data.empty:
            return pd.DataFrame()

        mask = self.hkex_data['stock_name'].str.contains(
            name_query, case=False, na=False
        )
        return self.hkex_data[mask][['stock_code', 'stock_name']].drop_duplicates()

    def get_all_stocks(self) -> pd.DataFrame:
        """
        Get all unique stocks from HKEX data.

        Returns:
            DataFrame with unique stock_code and stock_name pairs.
        """
        if self.hkex_data is None or self.hkex_data.empty:
            return pd.DataFrame()

        return self.hkex_data[['stock_code', 'stock_name']].drop_duplicates()

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file using pdfplumber with PyPDF2 fallback.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("PDF file not found: %s", pdf_path)
            return ""

        text = self._extract_with_pdfplumber(pdf_path)

        if not text.strip():
            logger.debug("pdfplumber returned no text, trying PyPDF2 for: %s", pdf_path.name)
            text = self._extract_with_pypdf2(pdf_path)

        return text

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts: List[str] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.debug("pdfplumber error on page: %s", e)
        except Exception as e:
            logger.warning("pdfplumber failed to open %s: %s", pdf_path.name, e)
        return "\n".join(text_parts)

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 as a fallback."""
        text_parts: List[str] = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.debug("PyPDF2 error on page: %s", e)
        except Exception as e:
            logger.error("PyPDF2 failed to read %s: %s", pdf_path.name, e)
        return "\n".join(text_parts)

    def get_available_pdfs(self, pdf_folder: Union[str, Path]) -> List[str]:
        """
        Get list of stock codes that have corresponding PDF files.

        Args:
            pdf_folder: Path to the folder containing PDF files.

        Returns:
            List of stock codes (filenames without .pdf extension).
        """
        pdf_folder = Path(pdf_folder)
        if not pdf_folder.exists():
            return []

        return sorted([
            f.stem for f in pdf_folder.iterdir()
            if f.suffix.lower() == '.pdf'
        ])

    def save_json(self, data: Any, filepath: Union[str, Path]) -> None:
        """Save data as a JSON file."""
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved: %s", filepath)
        except Exception as e:
            logger.error("Error saving JSON to %s: %s", filepath, e)

    def load_json(self, filepath: Union[str, Path]) -> Optional[Any]:
        """Load data from a JSON file."""
        filepath = Path(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("JSON file not found: %s", filepath)
            return None
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", filepath, e)
            return None
        except Exception as e:
            logger.error("Error loading JSON from %s: %s", filepath, e)
            return None

    def save_csv(self, df: pd.DataFrame, filepath: Union[str, Path]) -> None:
        """Save DataFrame as a CSV file."""
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info("Saved: %s", filepath)
        except Exception as e:
            logger.error("Error saving CSV to %s: %s", filepath, e)

    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load a CSV file as a DataFrame."""
        filepath = Path(filepath)
        try:
            return pd.read_csv(filepath, encoding='utf-8-sig')
        except FileNotFoundError:
            logger.warning("CSV file not found: %s", filepath)
            return pd.DataFrame()
        except Exception as e:
            logger.error("Error loading CSV from %s: %s", filepath, e)
            return pd.DataFrame()