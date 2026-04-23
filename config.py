"""
ESG Analysis System Configuration
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================
# Paths Configuration
# ============================================
BASE_DIR = Path(__file__).parent
DATA_FOLDER = BASE_DIR / 'data'
PDF_FOLDER = DATA_FOLDER
OUTPUT_DIR = BASE_DIR / 'outputs'
JSON_DIR = OUTPUT_DIR / 'esg_texts'
VIZ_DIR = OUTPUT_DIR / 'visualizations'
TEMP_DATA_FOLDER = BASE_DIR / 'temp_uploads'
TEMP_JSON_DIR = TEMP_DATA_FOLDER / 'esg_texts'

# Create all directories in one place
REQUIRED_DIRS: List[Path] = [
    OUTPUT_DIR,
    JSON_DIR,
    VIZ_DIR,
    TEMP_DATA_FOLDER,
    TEMP_JSON_DIR,
]
for dir_path in REQUIRED_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

HKEX_DATA_FILE = BASE_DIR / 'extracted_hkex_data.csv'

# ============================================
# ESG Keywords Configuration
# ============================================
ESG_KEYWORDS: Dict[str, List[str]] = {
    'Environmental': [
        'carbon', 'emission', 'climate', 'energy', 'sustain', 'environment',
        'renewable', 'water', 'waste', 'pollution', 'green', 'eco', 'solar',
        'wind', 'footprint', 'biodiversity', 'conservation', 'resource',
        'efficiency', 'recycle', 'circular', 'net zero', 'emiss', 'environ',
        'climat', 'energi', 'emissions', 'sustainable', 'environmental',
        'ghg', 'greenhouse', 'scope 1', 'scope 2', 'scope 3', 'decarbonization'
    ],
    'Social': [
        'employee', 'divers', 'inclus', 'community', 'human', 'health',
        'safety', 'workplace', 'gender', 'equal', 'wellbeing', 'people',
        'staff', 'worker', 'labor', 'rights', 'philanthropy', 'volunteer',
        'education', 'training', 'development', 'diversity', 'inclusion',
        'social', 'human rights', 'community engagement', 'dei', 'equity',
        'fair', 'wage', 'benefit', 'turnover', 'retention', 'talent'
    ],
    'Governance': [
        'govern', 'ethic', 'compli', 'board', 'transpar', 'audit',
        'risk', 'policy', 'stakeholder', 'integrity', 'code',
        'conduct', 'accountability', 'oversight', 'remuneration',
        'shareholder', 'disclosure', 'anti-corruption', 'whistleblower',
        'governance', 'ethical', 'compliance', 'transparency',
        'independent director', 'executive', 'compensation', 'voting'
    ]
}

# ============================================
# Performance Metrics Patterns
# ============================================
METRIC_PATTERNS: List[Tuple[str, str]] = [
    # Carbon emissions
    (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(tonnes?|tons?|tCO2e?|MTCO2e?)\s*(?:of\s*)?(CO2|carbon|emissions?|GHG)', 'Carbon Emissions'),
    (r'(Scope\s*[123])\s*(?:emissions?)?[:\s]*(\d+(?:,\d{3})*(?:\.\d+)?)', 'Scope Emissions'),
    # Energy
    (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(MWh|GWh|kWh|MWh)', 'Energy Consumption'),
    (r'(\d+(?:\.\d+)?)\s*%\s*(renewable|clean)\s*energy', 'Renewable Energy Percentage'),
    # Water
    (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(liters?|gallons?|m3|cubic\s*meters?)\s*(?:of\s*)?water', 'Water Usage'),
    (r'(\d+(?:\.\d+)?)\s*%\s*water\s*(reduction|recycling|reuse)', 'Water Efficiency'),
    # Waste
    (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(tonnes?|tons?|kg)\s*(?:of\s*)?waste', 'Waste Generation'),
    (r'(\d+(?:\.\d+)?)\s*%\s*(recycled|diverted|recovered)', 'Waste Diversion Rate'),
    # Social metrics
    (r'(\d+(?:\.\d+)?)\s*%\s*(female|women|gender\s*diversity)', 'Gender Diversity'),
    (r'(\d+(?:\.\d+)?)\s*%\s*(turnover|retention)', 'Employee Turnover'),
    (r'(\d+(?:,\d{3})*)\s*(hours?|days?)\s*(training|development)', 'Training Hours'),
    # Safety metrics
    (r'(LTIFR|TRIR|Lost\s*Time\s*Injury)[:\s]*(\d+(?:\.\d+)?)', 'Safety Rate'),
]

# ============================================
# ESG Topics for Classification
# ============================================
ESG_TOPICS: Dict[str, List[str]] = {
    'Climate Policy': ['carbon', 'emission', 'net zero', 'climate change', 'ghg', 'decarbonization'],
    'Energy Efficiency': ['energy', 'renewable', 'efficiency', 'consumption', 'solar', 'wind'],
    'Waste Management': ['waste', 'recycle', 'circular', 'landfill', 'hazardous'],
    'Water Stewardship': ['water', 'conservation', 'withdrawal', 'discharge', 'effluent'],
    'Biodiversity': ['biodiversity', 'ecosystem', 'habitat', 'species', 'conservation'],
    'Employee Welfare': ['employee', 'worker', 'safety', 'wellbeing', 'health'],
    'Diversity & Inclusion': ['diversity', 'inclusion', 'gender', 'equality', 'dei'],
    'Community Engagement': ['community', 'philanthropy', 'volunteer', 'local'],
    'Human Rights': ['human rights', 'labor', 'forced labor', 'child labor'],
    'Talent Development': ['training', 'development', 'education', 'career'],
    'Board Composition': ['board', 'director', 'independent', 'committee'],
    'Risk Management': ['risk', 'compliance', 'audit', 'internal control'],
    'Business Ethics': ['ethics', 'integrity', 'anti-corruption', 'transparency'],
    'Shareholder Rights': ['shareholder', 'voting', 'proxy', 'activist'],
    'Executive Compensation': ['executive', 'compensation', 'remuneration', 'pay ratio']
}

# ============================================
# LDA Model Configuration (保留，可能被其他模块引用)
# ============================================
LDA_CONFIG: Dict[str, object] = {
    'max_features': 2000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2),
    'learning_method': 'online',
    'learning_decay': 0.7,
    'max_iter': 150,
    'random_state': 42
}

# ============================================
# Visualization Configuration
# ============================================
VIZ_CONFIG: Dict[str, object] = {
    'figure_dpi': 150,
    'heatmap_cmap': 'Greens',
    'wordcloud_width': 400,
    'wordcloud_height': 400,
    'wordcloud_max_words': 50
}

print("Configuration loaded successfully!")