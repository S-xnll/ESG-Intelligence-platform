"""
ESG Analysis Dashboard - Interactive Streamlit Application
Supports: PDF processing, saved data loading, new file uploads, and ESG analysis
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_FOLDER,
    HKEX_DATA_FILE,
    OUTPUT_DIR,
    JSON_DIR,
    TEMP_DATA_FOLDER,
    TEMP_JSON_DIR,
)
from modules import PDFExtractor, TopicClassifier
from utils import DataLoader, TextProcessor

logger = logging.getLogger(__name__)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="ESG Analysis System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

_CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1B5E20;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #FF9800;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #F44336;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #388E3C;
    }
    .upload-section {
        border: 2px dashed #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #F9FBE7;
    }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Helper: data check decorator
# ============================================================

def require_data(func: Callable) -> Callable:
    """Decorator that shows a warning if no statements_df is loaded."""

    def wrapper(self: "ESGDashboard", *args, **kwargs):
        if (
            st.session_state.get('statements_df') is None
            or st.session_state['statements_df'].empty
        ):
            st.warning(
                "No data loaded. Please go to 'Data Selection & Loading' page first."
            )
            return
        return func(self, *args, **kwargs)

    return wrapper


# ============================================================
# Abstract Page Base
# ============================================================

class BasePage(ABC):
    """Abstract base for a dashboard page."""

    def __init__(self, dashboard: "ESGDashboard") -> None:
        self.dashboard = dashboard

    @abstractmethod
    def render(self) -> None:
        """Render the page content."""
        ...

    @staticmethod
    def section_header(title: str) -> None:
        """Render a consistent section header."""
        st.markdown(f'<h2 class="sub-header">{title}</h2>', unsafe_allow_html=True)

    @staticmethod
    def info_box(message: str, box_type: str = "info") -> None:
        """Render a styled info/warning/success/error box."""
        st.markdown(
            f'<div class="{box_type}-box">{message}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Page Implementations
# ============================================================

class DataSelectionPage(BasePage):
    """Page: Data Selection & Loading."""

    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">📁 Data Selection & Loading</h1>',
            unsafe_allow_html=True,
        )

        selected_companies = self._render_company_selection()
        if not selected_companies:
            st.info("👆 Please select at least one company to proceed.")
            return

        st.markdown("---")
        self.section_header("Step 2: Choose Data Loading Method")

        loading_method = st.radio(
            "How would you like to load data for the selected companies?",
            [
                "📂 Load Saved Data (if available)",
                "🔄 Process Original PDF Files",
                "📤 Upload New Files (PDF/TXT) - Temp Data",
                "🔀 Mixed: Load Saved + Process New PDFs",
            ],
        )

        st.markdown("---")

        method_handlers = {
            "📂 Load Saved Data (if available)": self._load_saved_data_only,
            "🔄 Process Original PDF Files": self._process_pdf_files,
            "📤 Upload New Files (PDF/TXT) - Temp Data": self._upload_new_files,
            "🔀 Mixed: Load Saved + Process New PDFs": self._mixed_load,
        }
        method_handlers[loading_method](selected_companies)

        st.markdown("---")
        self.section_header("Currently Loaded Data")
        self._show_loaded_data()

    # ----- Company Selection -----

    def _render_company_selection(self) -> List[str]:
        """Render the company selection UI. Returns selected stock codes."""
        if st.session_state['pdf_extractor'] is None:
            st.session_state['pdf_extractor'] = PDFExtractor()

        available_df = st.session_state['pdf_extractor'].get_available_stocks()
        if available_df.empty:
            st.warning("No PDF files found in the data folder.")
            st.info(f"Place PDF files in: {DATA_FOLDER}")
            return []

        processed = self.dashboard.get_processed_stock_codes()
        temp_codes = self.dashboard.get_temp_stock_codes()

        def _status(code: str) -> str:
            if code in temp_codes:
                return '📁 Temp'
            if code in processed:
                return '✅ Saved'
            return '🆕 New'

        available_df['status'] = available_df['stock_code'].apply(_status)

        st.markdown(f"**{len(available_df)} PDF files available**")
        display_cols = ['stock_code', 'stock_name', 'release_time', 'status']
        show_cols = [c for c in display_cols if c in available_df.columns]
        st.dataframe(available_df[show_cols], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            select_method = st.radio(
                "Selection method:",
                ["Select All", "Select Specific", "Select by Name"],
                key="company_select_method",
            )
        with col2:
            show_filter = st.radio(
                "Show:",
                [
                    "All Companies",
                    "Only New (Unprocessed)",
                    "Only Saved (Processed)",
                    "Only Temp Data",
                ],
                key="show_filter",
            )

        # Filter
        filter_map = {
            "Only New (Unprocessed)": '🆕 New',
            "Only Saved (Processed)": '✅ Saved',
            "Only Temp Data": '📁 Temp',
        }
        if show_filter in filter_map:
            selectable = available_df[available_df['status'] == filter_map[show_filter]]
        else:
            selectable = available_df

        selectable_codes = selectable['stock_code'].tolist()

        if select_method == "Select All":
            selected = selectable_codes
            st.success(f"Selected {len(selected)} companies")
        elif select_method == "Select Specific":
            selected = st.multiselect(
                "Choose companies:",
                options=selectable_codes,
                default=[],
                format_func=lambda x: self.dashboard._get_company_display(x),
            )
        else:
            name_query = st.text_input("Enter company name keyword:")
            if name_query:
                matching = selectable[
                    selectable['stock_name'].str.contains(
                        name_query, case=False, na=False
                    )
                ]
                selected = matching['stock_code'].tolist()
                st.info(f"Found {len(selected)} matching companies")
            else:
                selected = []

        st.session_state['selected_companies'] = selected
        return selected

    # ----- Loading Handlers -----

    def _load_saved_data_only(self, selected: List[str]) -> None:
        """Load saved JSON data for the selected companies."""
        self.section_header("📂 Loading Saved Data")

        processed = self.dashboard.get_processed_stock_codes()
        temp_codes = self.dashboard.get_temp_stock_codes()
        all_avail = processed | temp_codes

        available = [c for c in selected if c in all_avail]
        missing = [c for c in selected if c not in all_avail]

        if missing:
            st.warning(f"No saved data for: {', '.join(missing)}")
        if not available:
            st.error("None of the selected companies have saved data.")
            return

        from_perm = [c for c in available if c in processed]
        from_temp = [c for c in available if c in temp_codes]
        if from_perm:
            st.info(f"📂 From permanent storage: {', '.join(from_perm)}")
        if from_temp:
            st.warning(
                f"📁 From temp storage (cleared on restart): {', '.join(from_temp)}"
            )

        if st.button("📥 Load Saved Data", type="primary"):
            with st.spinner(f"Loading {len(available)} companies..."):
                self.dashboard.load_companies_from_json(available)
            st.success(f"Loaded {len(available)} companies!")
            st.rerun()

    def _process_pdf_files(self, selected: List[str]) -> None:
        """Process original PDF files."""
        self.section_header("🔄 Processing Original PDF Files")

        col1, col2 = st.columns(2)
        with col1:
            min_len = st.slider(
                "Minimum sentence length:", 50, 200, 100, key="pdf_min_length"
            )
        with col2:
            force = st.checkbox("Force reprocess (ignore saved data)", value=False)

        if force:
            st.warning("⚠️ This will overwrite previously saved data.")

        processed = self.dashboard.get_processed_stock_codes()
        to_process = selected if force else [c for c in selected if c not in processed]
        already_done = [] if force else [c for c in selected if c in processed]

        if already_done:
            st.info(f"Already processed (skipping): {', '.join(already_done)}")
        if not to_process:
            st.success("All selected companies are already processed!")
            if st.button("📥 Load Saved Data Instead"):
                self.dashboard.load_companies_from_json(selected)
                st.rerun()
            return

        if st.button("🚀 Process PDFs", type="primary"):
            with st.spinner(f"Processing {len(to_process)} PDFs..."):
                self.dashboard.process_and_save_pdfs(to_process, min_len)
            st.success(f"Processed {len(to_process)} PDFs!")
            st.rerun()

    def _upload_new_files(self, selected: List[str]) -> None:
        """Handle upload of new files as temporary data."""
        self.section_header("📤 Upload New Files (Temporary Data)")

        st.markdown(
            """
        <div class="upload-section">
            <strong>📋 Instructions:</strong><br>
            1. Select the target company<br>
            2. Upload PDF or TXT files<br>
            3. Data goes to <strong>temporary storage</strong><br>
            4. <strong>⚠️ Temp data clears on restart!</strong>
        </div>
        """,
            unsafe_allow_html=True,
        )

        temp_codes = self.dashboard.get_temp_stock_codes()
        if temp_codes:
            with st.expander(f"📁 Current Temp Data ({len(temp_codes)} companies)"):
                for code in sorted(temp_codes):
                    st.write(f"- {code}")

        target = st.selectbox(
            "Select target company:",
            options=selected,
            format_func=lambda x: self.dashboard._get_company_display(x),
        )

        uploaded = st.file_uploader(
            "Upload PDF or TXT files:",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
        )

        if uploaded:
            st.markdown(f"**{len(uploaded)} file(s) for {target}**")
            file_details = [
                {'Filename': f.name, 'Type': f.type, 'Size (KB)': round(f.size / 1024, 1)}
                for f in uploaded
            ]
            st.dataframe(pd.DataFrame(file_details), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                min_len = st.slider(
                    "Minimum sentence length:",
                    50, 200, 100,
                    key="upload_min_length",
                )
            with col2:
                append_mode = st.radio(
                    "Processing mode:",
                    ["Append to temp data", "Replace temp data"],
                )

            if st.button("📤 Process Uploaded Files (Temp)", type="primary"):
                with st.spinner("Processing..."):
                    self.dashboard.process_uploaded_files(
                        target, uploaded, min_len, append_mode
                    )
                st.success(f"Processed {len(uploaded)} file(s) for {target}")
                st.rerun()

    def _mixed_load(self, selected: List[str]) -> None:
        """Mixed mode: load saved + process remaining."""
        self.section_header("🔀 Mixed Loading Mode")

        processed = self.dashboard.get_processed_stock_codes()
        temp_codes = self.dashboard.get_temp_stock_codes()
        all_avail = processed | temp_codes

        have_data = [c for c in selected if c in all_avail]
        need_process = [c for c in selected if c not in all_avail]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Have Data")
            for c in have_data:
                source = "📁 Temp" if c in temp_codes else "📂 Saved"
                st.success(f"{c} - {self.dashboard._get_company_name(c)} ({source})")
            if not have_data:
                st.info("No data available")
        with col2:
            st.markdown("### 🆕 Need Processing")
            for c in need_process:
                st.warning(f"{c} - {self.dashboard._get_company_name(c)}")
            if not need_process:
                st.success("All companies have data!")

        min_len = st.slider(
            "Minimum sentence length:", 50, 200, 100, key="mixed_min_length"
        )

        if st.button("🚀 Load Saved + Process New", type="primary"):
            with st.spinner("Processing..."):
                if have_data:
                    self.dashboard.load_companies_from_json(have_data)
                if need_process:
                    self.dashboard.process_and_save_pdfs(need_process, min_len)
            st.success(
                f"Done! Loaded {len(have_data)} + processed {len(need_process)} "
                f"= {len(have_data) + len(need_process)} total"
            )
            st.rerun()

    def _show_loaded_data(self) -> None:
        """Display a summary of data currently in memory."""
        df = st.session_state.get('statements_df')
        if df is None or df.empty:
            st.info("No data currently loaded in memory.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Statements", len(df))
        col2.metric("Companies", df['stock_code'].nunique())
        col3.metric("Avg Length", f"{df['statement'].str.len().mean():.0f} chars")

        st.markdown("**Data by Company:**")
        summary = (
            df.groupby('stock_code')
            .agg(stock_name=('stock_name', 'first'), statements=('statement', 'count'))
        )
        temp_codes = self.dashboard.get_temp_stock_codes()
        processed = self.dashboard.get_processed_stock_codes()

        def _src(code: str) -> str:
            if code in temp_codes:
                return '📁 Temp'
            if code in processed:
                return '📂 Saved'
            return '❓ Unknown'

        summary['source'] = summary.index.map(_src)
        st.dataframe(summary, use_container_width=True)

        if st.button("🗑️ Clear All Loaded Data", type="secondary"):
            st.session_state['extracted_data'] = None
            st.session_state['statements_df'] = None
            st.rerun()


class DescriptiveAnalyticsPage(BasePage):
    """Page: Descriptive Analytics."""

    @require_data
    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">📊 Descriptive Analytics</h1>',
            unsafe_allow_html=True,
        )

        df: pd.DataFrame = st.session_state['statements_df']
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "Summary Statistics",
                "Frequency Distribution",
                "Charts & Visualizations",
                "Text Statistics",
            ],
        )

        if analysis_type == "Summary Statistics":
            self._render_summary(df)
        elif analysis_type == "Frequency Distribution":
            self._render_frequency(df)
        elif analysis_type == "Charts & Visualizations":
            self._render_charts(df)
        elif analysis_type == "Text Statistics":
            self._render_text_stats(df)

    @staticmethod
    def _render_summary(df: pd.DataFrame) -> None:
        DescriptiveAnalyticsPage.section_header("Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Statements", len(df))
        col2.metric("Companies", df['stock_code'].nunique())
        avg_len = df['statement'].str.len().mean()
        col3.metric("Avg Statement Length", f"{avg_len:.0f} chars")
        total_chars = df['statement'].str.len().sum()
        col4.metric("Total Characters", f"{total_chars:,}")

        st.markdown("### Statistics by Company")
        company_stats = (
            df.groupby('stock_code')
            .agg(
                stock_name=('stock_name', 'first'),
                statement_count=('statement', 'count'),
            )
        )
        company_stats['avg_length'] = (
            df.groupby('stock_code')['statement']
            .apply(lambda x: x.str.len().mean())
        )
        st.dataframe(company_stats, use_container_width=True)

        st.markdown("### Descriptive Statistics")
        st.dataframe(
            df['statement'].str.len().describe().to_frame('length'),
            use_container_width=True,
        )

    @staticmethod
    def _render_frequency(df: pd.DataFrame) -> None:
        DescriptiveAnalyticsPage.section_header("Frequency Distribution")

        st.markdown("### Statements per Company")
        counts = df['stock_code'].value_counts().reset_index()
        counts.columns = ['Stock Code', 'Count']
        fig = px.bar(
            counts, x='Stock Code', y='Count',
            title="Statement Count by Company",
            color='Count', color_continuous_scale='Greens',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Statement Length Distribution")
        df_len = df.copy()
        df_len['length'] = df_len['statement'].str.len()
        fig = px.histogram(
            df_len, x='length', nbins=30,
            title="Statement Length Distribution",
            color_discrete_sequence=['#4CAF50'],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Statement Length by Company")
        fig = px.box(
            df_len, x='stock_code', y='length',
            title="Statement Length Distribution by Company",
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_charts(df: pd.DataFrame) -> None:
        DescriptiveAnalyticsPage.section_header("Charts & Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Pie Chart", "Histogram", "Box Plot", "Scatter Plot"],
        )

        df_chart = df.copy()
        df_chart['length'] = df_chart['statement'].str.len()
        df_chart['word_count'] = df_chart['statement'].str.split().str.len()

        if chart_type == "Bar Chart":
            counts = df_chart['stock_code'].value_counts().reset_index()
            counts.columns = ['Stock Code', 'Count']
            fig = px.bar(
                counts, x='Stock Code', y='Count',
                title="Statements per Company",
                color='Count', color_continuous_scale='Greens',
            )
        elif chart_type == "Pie Chart":
            counts = df_chart['stock_code'].value_counts().reset_index()
            counts.columns = ['Stock Code', 'Count']
            fig = px.pie(
                counts, values='Count', names='Stock Code',
                title="Statement Distribution by Company",
            )
        elif chart_type == "Histogram":
            fig = px.histogram(
                df_chart, x='length', nbins=30,
                title="Statement Length Histogram",
            )
        elif chart_type == "Box Plot":
            fig = px.box(
                df_chart, x='stock_code', y='length',
                title="Statement Length Box Plot",
            )
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                df_chart, x='length', y='word_count',
                color='stock_code', hover_data=['stock_name'],
                title="Length vs Word Count",
            )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_text_stats(df: pd.DataFrame) -> None:
        DescriptiveAnalyticsPage.section_header("Text Statistics")

        df_text = df.copy()
        df_text['char_count'] = df_text['statement'].str.len()
        df_text['word_count'] = df_text['statement'].str.split().str.len()
        df_text['avg_word_length'] = df_text['statement'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
        )

        stats = {
            'Mean Characters': df_text['char_count'].mean(),
            'Median Characters': df_text['char_count'].median(),
            'Std Characters': df_text['char_count'].std(),
            'Mean Words': df_text['word_count'].mean(),
            'Median Words': df_text['word_count'].median(),
            'Std Words': df_text['word_count'].std(),
            'Avg Word Length': df_text['avg_word_length'].mean(),
        }
        st.dataframe(
            pd.DataFrame(stats.items(), columns=['Metric', 'Value']),
            use_container_width=True,
        )

        st.markdown("### Sample Statements")
        st.dataframe(
            df_text[
                ['stock_code', 'statement', 'char_count', 'word_count', 'avg_word_length']
            ].head(20),
            use_container_width=True,
        )


class TopicClassificationPage(BasePage):
    """Page: Topic Classification."""

    @require_data
    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">🤖 Topic Classification</h1>',
            unsafe_allow_html=True,
        )

        df: pd.DataFrame = st.session_state['statements_df']
        sample_size = st.slider(
            "Sample size for classification:", 100, min(1000, len(df)), 500
        )

        if st.button("🚀 Run Topic Classification", type="primary"):
            with st.spinner("Classifying statements..."):
                classifier = TopicClassifier(statements_df=df)
                results = classifier.run_full_classification(sample_size=sample_size)
                st.session_state['classification_results'] = results
            st.success("Classification complete!")

            if 'predicted_topic' in results.columns:
                counts = (
                    results['predicted_topic'].value_counts().reset_index()
                )
                counts.columns = ['Topic', 'Count']
                fig = px.pie(
                    counts, values='Count', names='Topic',
                    title="Topic Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Sample Classifications")
                st.dataframe(
                    results[['stock_code', 'statement', 'predicted_topic']].head(20),
                    use_container_width=True,
                )

                st.markdown("### Company Topic Profile")
                profile = classifier.get_company_topic_profile()
                st.dataframe(profile, use_container_width=True)


class CorrelationAnalysisPage(BasePage):
    """Page: Correlation Analysis."""

    @require_data
    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">📈 Correlation Analysis</h1>',
            unsafe_allow_html=True,
        )

        df: pd.DataFrame = st.session_state['statements_df'].copy()
        df['statement_length'] = df['statement'].str.len()
        df['word_count'] = df['statement'].str.split().str.len()

        company_stats = (
            df.groupby('stock_code')
            .agg(
                statement_count=('statement_length', 'count'),
                avg_length=('statement_length', 'mean'),
                std_length=('statement_length', 'std'),
                total_length=('statement_length', 'sum'),
                avg_words=('word_count', 'mean'),
            )
            .round(2)
            .reset_index()
        )

        st.markdown("### Company Statistics")
        st.dataframe(company_stats, use_container_width=True)

        st.markdown("### Correlation Matrix")
        numeric_cols = company_stats.select_dtypes(include=[np.number]).columns
        corr = company_stats[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Scatter Plot Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox(
                "Y-axis:", numeric_cols,
                index=min(1, len(numeric_cols) - 1),
            )

        fig = px.scatter(
            company_stats, x=x_col, y=y_col,
            hover_data=['stock_code'],
            title=f"{y_col} vs {x_col}",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(company_stats) > 1:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            model = LinearRegression()
            X = company_stats[[x_col]].values
            y = company_stats[y_col].values
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))

            st.markdown(
                f"""
            <div class="info-box">
                <strong>Regression Summary:</strong><br>
                • Slope: {model.coef_[0]:.4f}<br>
                • Intercept: {model.intercept_:.4f}<br>
                • R² Score: {r2:.4f}
            </div>
            """,
                unsafe_allow_html=True,
            )


class ExportResultsPage(BasePage):
    """Page: Export Results."""

    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">💾 Export Results</h1>',
            unsafe_allow_html=True,
        )

        options = self._get_export_options()
        if not options:
            st.info("No data available for export. Run some analyses first.")
            return

        selected = st.multiselect("Choose data to export:", options)
        for item in selected:
            self._render_export_item(item)

        st.markdown("---")
        st.markdown("### Save All Results to Output Folder")
        if st.button("💾 Save All Results"):
            st.session_state.setdefault('analysis_history', []).append({
                'timestamp': datetime.now().isoformat(),
                'action': 'Save All Results',
            })
            self.dashboard.save_all_results()

    def _get_export_options(self) -> List[str]:
        options: List[str] = []
        ss = st.session_state
        if ss.get('statements_df') is not None and not ss['statements_df'].empty:
            options.append("Extracted Statements")
        if ss.get('extracted_data') is not None:
            options.append("ESG Summary Statistics")
        if ss.get('classification_results') is not None:
            options.append("Topic Classification Results")
        return options

    def _render_export_item(self, item: str) -> None:
        st.markdown(f"#### {item}")
        if item == "Extracted Statements":
            df = st.session_state['statements_df']
            self._download_button(df, item, "esg_statements")
            st.dataframe(df.head(5), use_container_width=True)

        elif item == "ESG Summary Statistics":
            summary = []
            for data in st.session_state.get('extracted_data', []):
                row = data['statistics'].copy()
                row['stock_code'] = data['stock_code']
                row['stock_name'] = data.get('stock_name', '')
                summary.append(row)
            df = pd.DataFrame(summary)
            self._download_button(df, item, "esg_summary")
            st.dataframe(df, use_container_width=True)

        elif item == "Topic Classification Results":
            df = st.session_state.get('classification_results')
            if df is not None:
                self._download_button(df, item, "classifications")
                st.dataframe(df.head(5), use_container_width=True)

    @staticmethod
    def _download_button(df: pd.DataFrame, label: str, prefix: str) -> None:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"📥 Download {label}",
            data=csv,
            file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


class AnalysisHistoryPage(BasePage):
    """Page: Analysis History."""

    def render(self) -> None:
        st.markdown(
            '<h1 class="main-header">📋 Analysis History</h1>',
            unsafe_allow_html=True,
        )

        history = st.session_state.get('analysis_history', [])
        if not history:
            st.info("No analysis history yet.")
            return

        st.dataframe(pd.DataFrame(history), use_container_width=True)
        if st.button("Clear History"):
            st.session_state['analysis_history'] = []
            st.rerun()


# ============================================================
# Main Dashboard Class
# ============================================================

class ESGDashboard:
    """Interactive ESG Analysis Dashboard."""

    # Page registry
    PAGES: Dict[str, type] = {
        "📁 Data Selection & Loading": DataSelectionPage,
        "📊 Descriptive Analytics": DescriptiveAnalyticsPage,
        "🤖 Topic Classification": TopicClassificationPage,
        "📈 Correlation Analysis": CorrelationAnalysisPage,
        "💾 Export Results": ExportResultsPage,
        "📋 Analysis History": AnalysisHistoryPage,
    }

    def __init__(self) -> None:
        self.data_loader = DataLoader()
        self.text_processor = TextProcessor()
        self.hkex_data = self.data_loader.load_hkex_data(HKEX_DATA_FILE)
        self._cleanup_temp_folder()
        self._init_session_state()

    # ============================================================
    # Session State
    # ============================================================

    @staticmethod
    def _init_session_state() -> None:
        defaults: Dict[str, Any] = {
            'extracted_data': None,
            'statements_df': None,
            'selected_companies': [],
            'classification_results': None,
            'analysis_history': [],
            'already_processed': set(),
            'new_files_processed': {},
            'temp_data_loaded': set(),
            'pdf_extractor': None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # ============================================================
    # Temp Folder
    # ============================================================

    @staticmethod
    def _cleanup_temp_folder() -> None:
        if TEMP_DATA_FOLDER.exists():
            for item in TEMP_DATA_FOLDER.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception:
                    logger.warning("Could not delete %s", item, exc_info=True)
        TEMP_JSON_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Temporary folder cleaned up.")

    # ============================================================
    # Sidebar
    # ============================================================

    def render_sidebar(self) -> str:
        """Render sidebar navigation. Returns the selected page name."""
        st.sidebar.image(
            "https://img.icons8.com/color/96/000000/sustainability.png",
            width=80,
        )
        st.sidebar.title("🌱 ESG Analysis")

        page = st.sidebar.radio("Select Function", list(self.PAGES.keys()))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Status")
        status = self._check_data_status()

        col1, col2, col3, col4 = st.sidebar.columns(4)
        col1.metric("PDFs", status['available_pdfs'])
        col2.metric("Done", status['processed'])
        col3.metric("Temp", status['temp_data'])
        col4.metric("New", status['unprocessed'])

        ss_df = st.session_state.get('statements_df')
        if ss_df is not None and not ss_df.empty:
            st.sidebar.success(f"✅ {len(ss_df)} statements loaded")
        else:
            st.sidebar.info("No data in memory")

        temp_codes = self.get_temp_stock_codes()
        if temp_codes:
            with st.sidebar.expander(f"📁 Temp Data ({len(temp_codes)})"):
                for code in sorted(temp_codes):
                    st.write(f"- {code}")
                st.warning("Temp data clears on restart")

        if st.sidebar.button("🔄 Refresh Status"):
            st.rerun()

        if temp_codes and st.sidebar.button("🗑️ Clear Temp Data Now", type="secondary"):
            self._cleanup_temp_folder()
            st.session_state['temp_data_loaded'] = set()
            st.rerun()

        return page

    # ============================================================
    # Data Status
    # ============================================================

    @staticmethod
    def get_processed_stock_codes() -> Set[str]:
        """Return set of stock codes with permanent JSON files."""
        if not JSON_DIR.exists():
            return set()
        return {
            f.stem.replace('_esg_texts', '')
            for f in JSON_DIR.iterdir()
            if f.suffix == '.json' and '_esg_texts' in f.stem
        }

    @staticmethod
    def get_temp_stock_codes() -> Set[str]:
        """Return set of stock codes with temporary JSON files."""
        if not TEMP_JSON_DIR.exists():
            return set()
        return {
            f.stem.replace('_esg_texts', '')
            for f in TEMP_JSON_DIR.iterdir()
            if f.suffix == '.json' and '_esg_texts' in f.stem
        }

    def _check_data_status(self) -> Dict[str, Any]:
        processed = self.get_processed_stock_codes()
        temp_codes = self.get_temp_stock_codes()
        available = set(self.data_loader.get_available_pdfs(DATA_FOLDER))
        return {
            'available_pdfs': len(available),
            'processed': len(processed),
            'temp_data': len(temp_codes),
            'unprocessed': len(available - processed),
        }

    # ============================================================
    # JSON I/O
    # ============================================================

    def load_single_json(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """Load a single ESG text JSON (temp first, then permanent)."""
        for folder, source in [(TEMP_JSON_DIR, 'temp'), (JSON_DIR, 'permanent')]:
            filepath = folder / f"{stock_code}_esg_texts.json"
            if filepath.exists():
                try:
                    data = json.loads(filepath.read_text(encoding='utf-8'))
                    data['_source'] = source
                    return data
                except Exception:
                    logger.warning("Error reading %s", filepath, exc_info=True)
        return None

    @staticmethod
    def save_esg_data(esg_data: Dict[str, Any], is_temp: bool = False) -> None:
        """Save ESG data dict to JSON."""
        folder = TEMP_JSON_DIR if is_temp else JSON_DIR
        stock_code = esg_data['stock_code']
        filepath = folder / f"{stock_code}_esg_texts.json"
        save_data = {k: v for k, v in esg_data.items() if not k.startswith('_')}
        DataLoader().save_json(save_data, filepath)

    # ============================================================
    # Company Helpers
    # ============================================================

    def _get_company_name(self, stock_code: str) -> str:
        info = self.data_loader.get_stock_info(stock_code)
        return info.get('stock_name', stock_code)

    def _get_company_display(self, stock_code: str) -> str:
        return f"{stock_code} - {self._get_company_name(stock_code)}"

    # ============================================================
    # Loading Companies from JSON
    # ============================================================

    def load_companies_from_json(self, stock_codes: List[str]) -> None:
        """Load multiple companies from saved JSON into session state."""
        existing_data: List[Dict] = (
            list(st.session_state['extracted_data'])
            if st.session_state['extracted_data']
            else []
        )
        all_statements: List[Dict] = []

        # Keep companies not in this load batch
        existing_data = [d for d in existing_data if d['stock_code'] not in stock_codes]
        existing_statements = []
        ss_df = st.session_state.get('statements_df')
        if ss_df is not None and not ss_df.empty:
            existing_statements = ss_df[
                ~ss_df['stock_code'].isin(stock_codes)
            ].to_dict('records')

        for code in stock_codes:
            esg_data = self.load_single_json(code)
            if esg_data is None:
                continue
            existing_data.append(esg_data)
            for sentence in esg_data.get('all_sentences', []):
                lemma = self.text_processor.lemmatize_text(sentence)
                if len(lemma) > 50:
                    all_statements.append({
                        'stock_code': code,
                        'stock_name': esg_data.get('stock_name', code),
                        'statement': sentence,
                        'lemma': lemma,
                    })

        all_statements.extend(existing_statements)
        st.session_state['extracted_data'] = existing_data
        st.session_state['statements_df'] = (
            pd.DataFrame(all_statements) if all_statements else pd.DataFrame()
        )

    # ============================================================
    # Processing PDFs
    # ============================================================

    def process_and_save_pdfs(
        self, stock_codes: List[str], min_sentence_length: int = 100
    ) -> None:
        """Process PDFs and save results to permanent storage."""
        if st.session_state['pdf_extractor'] is None:
            st.session_state['pdf_extractor'] = PDFExtractor()

        existing_data: List[Dict] = (
            list(st.session_state['extracted_data'])
            if st.session_state['extracted_data']
            else []
        )
        all_statements: List[Dict] = []

        existing_data = [d for d in existing_data if d['stock_code'] not in stock_codes]
        ss_df = st.session_state.get('statements_df')
        existing_statements = []
        if ss_df is not None and not ss_df.empty:
            existing_statements = ss_df[
                ~ss_df['stock_code'].isin(stock_codes)
            ].to_dict('records')

        progress = st.progress(0)
        status_text = st.empty()
        total = len(stock_codes)

        for i, code in enumerate(stock_codes):
            status_text.text(f"Processing: {code} ({i + 1}/{total})")
            esg_data = st.session_state['pdf_extractor'].process_single_pdf(code)

            if esg_data:
                existing_data.append(esg_data)
                self.save_esg_data(esg_data, is_temp=False)

                for sentence in esg_data.get('all_sentences', []):
                    if len(sentence) >= min_sentence_length:
                        lemma = self.text_processor.lemmatize_text(sentence)
                        if len(lemma) > 50:
                            all_statements.append({
                                'stock_code': code,
                                'stock_name': esg_data.get('stock_name', code),
                                'statement': sentence,
                                'lemma': lemma,
                            })
            progress.progress((i + 1) / total)

        status_text.text("Processing complete!")
        all_statements.extend(existing_statements)
        st.session_state['extracted_data'] = existing_data
        st.session_state['statements_df'] = (
            pd.DataFrame(all_statements) if all_statements else pd.DataFrame()
        )
        st.session_state.setdefault('analysis_history', []).append({
            'timestamp': datetime.now().isoformat(),
            'action': 'PDF Processing',
            'companies': total,
            'statements_added': len(all_statements) - len(existing_statements),
        })

    # ============================================================
    # Processing Uploaded Files
    # ============================================================

    def process_uploaded_files(
        self,
        stock_code: str,
        uploaded_files: List[Any],
        min_sentence_length: int,
        append_mode: str,
    ) -> None:
        """
        Process uploaded PDF/TXT files and save to temporary storage.

        Args:
            stock_code: Target stock code.
            uploaded_files: List of Streamlit UploadedFile objects.
            min_sentence_length: Minimum sentence length.
            append_mode: "Append to temp data" or "Replace temp data".
        """
        all_text = ""
        for uf in uploaded_files:
            text = self._extract_text_from_uploaded_file(uf)
            if text:
                all_text += text + "\n\n"

        if not all_text.strip():
            st.error("No text could be extracted.")
            return

        esg_data, new_statements = self._process_extracted_text(
            all_text, stock_code, min_sentence_length=min_sentence_length
        )
        if esg_data is None:
            st.error("Failed to extract meaningful content.")
            return

        # Merge with existing temp data if appending
        if append_mode == "Append to temp data":
            existing = None
            temp_file = TEMP_JSON_DIR / f"{stock_code}_esg_texts.json"
            if temp_file.exists():
                try:
                    existing = json.loads(temp_file.read_text(encoding='utf-8'))
                except Exception:
                    pass

            if existing:
                old_sentences = set(existing['all_sentences'])
                new_sentences = [
                    s for s in esg_data['all_sentences']
                    if s not in old_sentences
                ]
                esg_data['all_sentences'] = existing['all_sentences'] + new_sentences

                for cat in esg_data['esg_texts']:
                    merged = set(existing['esg_texts'].get(cat, []))
                    merged.update(esg_data['esg_texts'].get(cat, []))
                    esg_data['esg_texts'][cat] = list(merged)

                esg_data['statistics']['total_sentences'] = len(esg_data['all_sentences'])
                esg_data['statistics']['esg_sentences'] = sum(
                    len(v) for v in esg_data['esg_texts'].values()
                )

        self.save_esg_data(esg_data, is_temp=True)
        st.session_state['temp_data_loaded'].add(stock_code)

        # Update statements DataFrame
        self._update_statements_for_company(
            stock_code, esg_data, new_statements,
            replace_mode=(append_mode == "Replace temp data"),
        )

        st.session_state.setdefault('new_files_processed', {}).setdefault(
            stock_code, []
        ).extend(f.name for f in uploaded_files)

    def _extract_text_from_uploaded_file(self, uploaded_file: Any) -> str:
        """Extract text from a single uploaded PDF or TXT file."""
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if suffix.lower() == '.pdf':
                return self.data_loader.extract_text_from_pdf(tmp_path)
            elif suffix.lower() == '.txt':
                return Path(tmp_path).read_text(encoding='utf-8')
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _process_extracted_text(
        self,
        text: str,
        stock_code: str,
        stock_name: Optional[str] = None,
        min_sentence_length: int = 100,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        """Build ESG data dict and statements list from raw text."""
        if not text.strip():
            return None, []

        if st.session_state['pdf_extractor'] is None:
            st.session_state['pdf_extractor'] = PDFExtractor()
        extractor = st.session_state['pdf_extractor']

        cleaned = self.text_processor.clean_text(text)
        sentences = self.text_processor.extract_sentences(
            cleaned, min_length=min_sentence_length
        )
        if not sentences:
            return None, []

        esg_sentences = extractor.extract_esg_sentences(sentences)
        esg_count = sum(len(v) for v in esg_sentences.values())

        if stock_name is None:
            stock_name = self._get_company_name(stock_code)

        esg_data = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'file_name': f"{stock_code}.pdf",
            'extraction_date': datetime.now().isoformat(),
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

        statements = []
        for sentence in sentences:
            if len(sentence) >= min_sentence_length:
                lemma = self.text_processor.lemmatize_text(sentence)
                if len(lemma) > 50:
                    statements.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'statement': sentence,
                        'lemma': lemma,
                    })

        return esg_data, statements

    @staticmethod
    def _update_statements_for_company(
        stock_code: str,
        esg_data: Dict[str, Any],
        new_statements: List[Dict[str, str]],
        replace_mode: bool,
    ) -> None:
        """Update the session-state statements DataFrame for one company."""
        existing_df = st.session_state.get('statements_df')
        existing_df = existing_df if existing_df is not None else pd.DataFrame()

        if not existing_df.empty and replace_mode:
            existing_df = existing_df[existing_df['stock_code'] != stock_code]

        updated_df = pd.concat(
            [existing_df, pd.DataFrame(new_statements)], ignore_index=True
        )
        st.session_state['statements_df'] = updated_df

        all_data = list(st.session_state.get('extracted_data') or [])
        if replace_mode:
            all_data = [d for d in all_data if d['stock_code'] != stock_code]
        all_data.append(esg_data)
        st.session_state['extracted_data'] = all_data

    # ============================================================
    # Save All Results
    # ============================================================

    def save_all_results(self) -> None:
        """Save all in-memory results to the output folder."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        saved = []

        ss_df = st.session_state.get('statements_df')
        if ss_df is not None and not ss_df.empty:
            path = OUTPUT_DIR / 'statements.csv'
            ss_df.to_csv(path, index=False, encoding='utf-8-sig')
            saved.append(str(path))

        class_results = st.session_state.get('classification_results')
        if class_results is not None:
            path = OUTPUT_DIR / 'topic_classifications.csv'
            class_results.to_csv(path, index=False, encoding='utf-8-sig')
            saved.append(str(path))

        if saved:
            st.success(f"Saved {len(saved)} files to {OUTPUT_DIR}")
            for f in saved:
                st.write(f"- {f}")

    # ============================================================
    # Run
    # ============================================================

    def run(self) -> None:
        """Main entry point — render sidebar then delegate to the chosen page."""
        page_name = self.render_sidebar()
        page_cls = self.PAGES.get(page_name)
        if page_cls is None:
            st.error(f"Unknown page: {page_name}")
            return

        page = page_cls(self)
        page.render()


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    dashboard = ESGDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()