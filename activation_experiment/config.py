"""
Activation Differential Experiment — Configuration

All hyperparameters and paths in one place.
"""

import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Model A = GLM (extracted to memmap in pass1)
# Model B = Llama (run live in pass2)
MODEL_A_ID = "THUDM/glm-4-9b-chat"
MODEL_A_FALLBACKS = ["THUDM/glm-4-9b"]
MODEL_A_NAME = "glm4_9b"
MODEL_A_LAYERS = 40   # transformer layers (hidden_states has 41: embedding + 40)
MODEL_A_HIDDEN = 4096
CORPUS_POSITIONS_KEY_A = "model_a_positions"
CORPUS_TOKEN_COUNT_KEY_A = "model_a_token_count"

MODEL_B_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # ungated mirror, same weights
MODEL_B_FALLBACKS = [
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
MODEL_B_NAME = "llama3.1_8b"
MODEL_B_LAYERS = 32   # transformer layers (hidden_states has 33: embedding + 32)
MODEL_B_HIDDEN = 4096
CORPUS_POSITIONS_KEY_B = "model_b_positions"
CORPUS_TOKEN_COUNT_KEY_B = "model_b_token_count"

HIDDEN_DIM = 4096  # shared hidden dimension

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
CORPUS_SIZE = 10_000
CORPUS_CATEGORIES = {
    "english_web": 3000,
    "code": 2000,
    "conversational": 2000,
    "math_reasoning": 1000,
    "multilingual": 1000,
    "mixed_edge": 1000,
}
MAX_TOKENS = 128
MIN_TOKENS = 16

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
CHECKPOINT_INTERVAL = 500   # save checkpoint every N inputs
PROGRESS_INTERVAL = 100     # print progress every N inputs

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
RANK_THRESHOLD = 0.95       # cumulative variance threshold for rank
TOP_K_PCS = 20              # principal components for the atlas
ATLAS_EXTREMES = 50         # extreme tokens per PC end
CKA_SUBSET_SIZE = 500       # inputs used for layer correspondence discovery
CATEGORY_LAYERS = 8         # number of layers for per-category covariance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIVATION_DIR = os.path.join(BASE_DIR, "activations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OVERALL_STATS_DIR = os.path.join(RESULTS_DIR, "overall_stats")
CATEGORY_STATS_DIR = os.path.join(RESULTS_DIR, "category_stats")
ATLAS_DIR = os.path.join(RESULTS_DIR, "atlas")

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
ACTIVATION_INDEX_PATH = os.path.join(DATA_DIR, "activation_index.json")
MODEL_A_MMAP_PATH = os.path.join(ACTIVATION_DIR, "model_a_activations.mmap")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoint.json")

LAYER_CORRESPONDENCE_PATH = os.path.join(RESULTS_DIR, "layer_correspondence.npy")
LAYER_MAPPING_PATH = os.path.join(RESULTS_DIR, "layer_mapping.json")

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for d in [DATA_DIR, ACTIVATION_DIR, RESULTS_DIR, OVERALL_STATS_DIR,
          CATEGORY_STATS_DIR, ATLAS_DIR]:
    os.makedirs(d, exist_ok=True)

for cat in CORPUS_CATEGORIES:
    os.makedirs(os.path.join(CATEGORY_STATS_DIR, cat), exist_ok=True)
