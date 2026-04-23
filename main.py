"""
ESG Analysis System - Main Execution File
Run all analysis modules in sequence or selectively
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import PDF_FOLDER, OUTPUT_DIR
from modules import (
    PDFExtractor,
    LDAModeler,
    KeywordExtractor,
    TopicClassifier,
    SentimentAnalyzer,
    AdvancedAnalytics,
    ESGPredictor,
    ESGChatbot
)


class ESGAnalysisPipeline:
    """Complete ESG Analysis Pipeline"""
    
    def __init__(self, pdf_folder=PDF_FOLDER, output_dir=OUTPUT_DIR):
        self.pdf_folder = pdf_folder
        self.output_dir = output_dir
        
        self.statements_df = None
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_module(self, module_name, **kwargs):
        """
        Run a specific module
        
        Args:
            module_name (str): Name of module to run
            **kwargs: Additional arguments for the module
        
        Returns:
            object: Module results
        """
        print("\n" + "=" * 80)
        print(f"RUNNING MODULE: {module_name.upper()}")
        print("=" * 80)
        
        if module_name == 'extract':
            extractor = PDFExtractor(pdf_folder=self.pdf_folder, output_dir=self.output_dir)
            all_data, statements_df = extractor.process_all_pdfs()
            self.statements_df = statements_df
            self.results['extract'] = {'all_data': all_data, 'statements_df': statements_df}
            return all_data, statements_df
        
        elif module_name == 'lda':
            modeler = LDAModeler(statements_df=self.statements_df, output_dir=self.output_dir)
            results = modeler.run_full_analysis()
            self.results['lda'] = results
            return results
        
        elif module_name == 'keywords':
            extractor = KeywordExtractor(statements_df=self.statements_df, output_dir=self.output_dir)
            results = extractor.run_full_extraction(**kwargs)
            self.results['keywords'] = results
            return results
        
        elif module_name == 'classify':
            classifier = TopicClassifier(statements_df=self.statements_df, output_dir=self.output_dir)
            results = classifier.run_full_classification(**kwargs)
            self.results['classify'] = results
            return results
        
        elif module_name == 'sentiment':
            analyzer = SentimentAnalyzer(statements_df=self.statements_df, output_dir=self.output_dir)
            results = analyzer.run_full_analysis(**kwargs)
            self.results['sentiment'] = results
            return results
        
        elif module_name == 'advanced':
            analytics = AdvancedAnalytics(statements_df=self.statements_df, output_dir=self.output_dir)
            results = analytics.run_full_analysis(**kwargs)
            self.results['advanced'] = results
            return results
        
        elif module_name == 'predict':
            predictor = ESGPredictor(output_dir=self.output_dir)
            results = predictor.run_full_prediction(**kwargs)
            self.results['predict'] = results
            return results
        
        elif module_name == 'chatbot':
            chatbot = ESGChatbot(data_path=self.output_dir)
            return chatbot
        
        elif module_name == 'dashboard':
            print("\nLaunching Streamlit Dashboard...")
            print("Run the following command in terminal:")
            print("  streamlit run dashboard.py")
            return None
        
        else:
            print(f"Unknown module: {module_name}")
            print("Available modules: extract, lda, keywords, classify, sentiment, advanced, predict, chatbot, dashboard")
            return None
    
    def run_all(self, skip_modules=None):
        """
        Run all modules in sequence
        
        Args:
            skip_modules (list): List of modules to skip
        
        Returns:
            dict: All results
        """
        if skip_modules is None:
            skip_modules = []
        
        modules = ['extract', 'lda', 'keywords', 'classify', 'sentiment', 'advanced', 'predict']
        
        print("\n" + "=" * 80)
        print("ESG ANALYSIS SYSTEM - FULL PIPELINE")
        print("=" * 80)
        print(f"PDF Folder: {self.pdf_folder}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Skipping modules: {skip_modules if skip_modules else 'None'}")
        print("=" * 80)
        
        for module in modules:
            if module in skip_modules:
                print(f"\n⏭ Skipping module: {module}")
                continue
            
            try:
                self.run_module(module)
            except Exception as e:
                print(f"\n❌ Error in module {module}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 80)
        print("✅ FULL PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved in: {self.output_dir}")
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print analysis summary"""
        print("\n" + "-" * 40)
        print("ANALYSIS SUMMARY")
        print("-" * 40)
        
        # Check output files
        output_files = os.listdir(self.output_dir)
        csv_files = [f for f in output_files if f.endswith('.csv')]
        json_files = [f for f in output_files if f.endswith('.json')]
        png_files = [f for f in os.listdir(os.path.join(self.output_dir, 'visualizations')) 
                    if f.endswith('.png')] if os.path.exists(os.path.join(self.output_dir, 'visualizations')) else []
        
        print(f"\nGenerated Files:")
        print(f"  📊 CSV files: {len(csv_files)}")
        print(f"  📄 JSON files: {len(json_files)}")
        print(f"  🖼️  Visualization files: {len(png_files)}")
        
        print("\n" + "=" * 40)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ESG Analysis System')
    parser.add_argument('--module', '-m', type=str, default='all',
                       choices=['all', 'extract', 'lda', 'keywords', 'classify', 
                               'sentiment', 'advanced', 'predict', 'chatbot', 'dashboard'],
                       help='Module to run')
    parser.add_argument('--pdf-folder', '-p', type=str, default=str(PDF_FOLDER),
                       help='Path to PDF folder')
    parser.add_argument('--output-dir', '-o', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--skip', '-s', nargs='+', default=[],
                       help='Modules to skip when running all')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for classification/sentiment')
    parser.add_argument('--use-spacy', action='store_true',
                       help='Use spaCy for keyword extraction')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ESGAnalysisPipeline(
        pdf_folder=args.pdf_folder,
        output_dir=args.output_dir
    )
    
    # Run selected module
    if args.module == 'all':
        pipeline.run_all(skip_modules=args.skip)
    elif args.module == 'dashboard':
        print("\n" + "=" * 60)
        print("LAUNCHING ESG DASHBOARD")
        print("=" * 60)
        print("\nRun the following command to start the dashboard:")
        print("\n  streamlit run dashboard.py")
        print("\nOr if you're in the esg_analysis_system directory:")
        print("\n  python -m streamlit run dashboard.py")
        print("\n" + "=" * 60)
    else:
        kwargs = {}
        if args.module in ['classify', 'sentiment'] and args.sample_size:
            kwargs['sample_size'] = args.sample_size
        if args.module == 'keywords' and args.use_spacy:
            kwargs['use_spacy'] = True
        
        pipeline.run_module(args.module, **kwargs)


if __name__ == "__main__":
    main()