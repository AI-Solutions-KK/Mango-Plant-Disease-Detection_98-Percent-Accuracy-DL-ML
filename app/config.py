# ================================
# FILE: app/config.py
# ================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CACHE_DIR = PROJECT_ROOT / "embeddings_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---- model artifacts ----
CLF_FILE = CACHE_DIR / "svc_model.pkl"
CENTROIDS_FILE = CACHE_DIR / "centroids.npy"
CLASSES_FILE   = CACHE_DIR / "classes.npy"

# ---- image settings ----
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (224, 224)

# ---- model settings ----
MODEL_NAME = "efficientnet_b4"
NUM_CLASSES = 8
