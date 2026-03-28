import os

# =========================================================
# Paths
# =========================================================
#BASE_DIR = "/home/your_username/SmartFall/prepared_smartfall_watch"
BASE_DIR = "/home/Students/stg60/RL/RL_Pole/smfallData/prepared_smartfall_meta_wrist"
OUTPUT_ROOT = "/home/Students/stg60/RL/RL_Pole/smfallData/results"

# =========================================================
# Training settings
# =========================================================
N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# =========================================================
# Model selection
# Change this only
# =========================================================
MODEL_NAME = "bilstm"   # options: "bilstm", "transformer"

# =========================================================
# Transformer settings
# =========================================================
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DROPOUT = 0.1

# =========================================================
# BiLSTM settings
# =========================================================
BILSTM_UNITS_1 = 64
BILSTM_UNITS_2 = 32
BILSTM_DROPOUT = 0.3


def get_model_output_dir(model_name: str) -> str:
    return os.path.join(OUTPUT_ROOT, f"{model_name}_results")
