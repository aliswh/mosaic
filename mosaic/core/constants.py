"""Constants used across the MOSAIC package."""

# VLLM constants
LLM_KWARGS = {
    'gpu_memory_utilization': 0.9,
    'max_model_len': 2048,
}

SAMPLING_KWARGS = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 64,
    'seed': 42,
    'max_tokens': 1024,
}

# Paths
SRC_PATH = "/home/alice/work/mosaic"

# Language codes
FB_TARGET_LANGUAGES = [
    "eng_Latn",  # English 
    "dan_Latn",  # Danish
    "spa_Latn",  # Spanish
    "fra_Latn",  # French
    "deu_Latn",  # German
    "ita_Latn",  # Italian
    "por_Latn",  # Portuguese
]

LANG_CODES = {
    # Germanic
    'eng_Latn': 'English',
    'deu_Latn': 'German', 
    'nld_Latn': 'Dutch',
    'dan_Latn': 'Danish',
    'nor_Latn': 'Norwegian',
    'swe_Latn': 'Swedish',
    
    # Romance
    'fra_Latn': 'French',
    'ita_Latn': 'Italian',
    'por_Latn': 'Portuguese', 
    'spa_Latn': 'Spanish',
    'ron_Latn': 'Romanian',
    
    # Slavic
    'pol_Latn': 'Polish',
    'ces_Latn': 'Czech',
    'slk_Latn': 'Slovak',
    'slv_Latn': 'Slovene',
    'bul_Cyrl': 'Bulgarian',
    'hrv_Latn': 'Croatian',
    
    # Baltic
    'lit_Latn': 'Lithuanian',
    'lav_Latn': 'Latvian',
    'est_Latn': 'Estonian',
    
    # Other Indo-European
    'ell_Grek': 'Greek',
    'gle_Latn': 'Irish',
    'hun_Latn': 'Hungarian',
    'fin_Latn': 'Finnish',
    'mlt_Latn': 'Maltese'
}