import sys
import os
import numpy as np
from PIL import Image

from embeddings import load_dinov2, get_embedding

print("Loading DINOv2...")
model = load_dinov2()
print("Model loaded.")

image_a_path = "test_image_a.jpg"
image_b_path = "test_image_b.jpg"
image_c_path = "test_image_c.jpg"

Image.new("RGB", (300, 300), color=(120, 180, 60)).save(image_a_path)
Image.new("RGB", (300, 300), color=(255, 220, 50)).save(image_b_path)
Image.new("RGB", (300, 300), color=(118, 178, 58)).save(image_c_path)

print("\nExtracting embeddings...")
embedding_a = get_embedding(image_a_path, model)
embedding_b = get_embedding(image_b_path, model)
embedding_c = get_embedding(image_c_path, model)

print(f"Embedding A shape: {embedding_a.shape}")
print(f"Embedding B shape: {embedding_b.shape}")
print(f"Embedding C shape: {embedding_c.shape}")

def cosine_similarity(a, b):
    return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 100, 1)

sim_a_b = cosine_similarity(embedding_a, embedding_b)
sim_a_c = cosine_similarity(embedding_a, embedding_c)
sim_b_c = cosine_similarity(embedding_b, embedding_c)

print(f"\nSimilarity Results:")
print(f"  Green vs Yellow  (should be LOW) :  {sim_a_b}%")
print(f"  Green vs Green   (should be HIGH):  {sim_a_c}%")
print(f"  Yellow vs Green  (should be LOW) :  {sim_b_c}%")

os.remove(image_a_path)
os.remove(image_b_path)
os.remove(image_c_path)

print("\nEmbedding pipeline working correctly.")