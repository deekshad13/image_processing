"""Train the fine-tuned CropSimilarityModel. Wrapper around src.model.fine_tune.train()."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model.fine_tune import train

if __name__ == "__main__":
    train()
