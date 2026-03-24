import os

BASE_DIR = os.environ.get("APP_BASE_DIR", r"D:\image_processing")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")