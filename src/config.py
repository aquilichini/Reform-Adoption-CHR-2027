from pathlib import Path

# ============================================================================
# Project paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RULES_FILE = DATA_DIR / "rules.csv"

PREPROCESSED_CACHE_JSON = OUTPUT_DIR / "preprocessed_conllu_corpus.json"
CLASSIFIED_CACHE = OUTPUT_DIR / "classified_conllu_corpus.json"

MODEL_CACHE_DIR = PROJECT_ROOT / "models"

FULL_CORPUS = DATA_DIR / "<set_directory_here>"
USE_TEST_CORPUS = False
CORPUS_DIR = FULL_CORPUS
AGGREGATION_MONTHS = 6
USE_EWMA = True
EWMA_SPAN = 4
SHOW_APPROX_LINE = True
APPROX_LINE_WINDOW = 9
ADOPTION_THRESHOLDS = (0.8, 0.9, 0.95)
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0
PLOT_N_JOBS = -1


LANGUAGE_LABEL_GROUPS = {
    "bokmaal": {"nob"},
    "bokmaal_dan": {"nob", "dan"},
    "nynorsk": {"nno"},
    "both": {"nob", "nno"},
    "both_dan": {"nob", "nno", "dan"},
}

RULE_LANGUAGE_TO_GROUP = {"nb": "bokmaal_dan", "nn": "nynorsk"}
NORMATIVE_EFFECT_GROUPS = {
    "main_to_main": {
        "no_status_change", "main_to_main", "main_to_main_equal",
    },
    "side_to_main": {"side_to_main"},
    "main_to_optional": {"main_to_optional", "main_to_side"},
}
