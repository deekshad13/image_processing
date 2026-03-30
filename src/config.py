import os

# Base directory — resolves to wherever the project is running from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_RAW_DIR        = os.getenv("DATA_RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
DATA_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")

# Model paths
FINE_TUNED_MODEL    = os.path.join(DATA_EMBEDDINGS_DIR, "fine_tuned_model.pt")

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Image validation settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))
MIN_BRIGHTNESS   = int(os.getenv("MIN_BRIGHTNESS", 30))
MAX_BRIGHTNESS   = int(os.getenv("MAX_BRIGHTNESS", 240))
MIN_LAPLACIAN    = int(os.getenv("MIN_LAPLACIAN", 20))