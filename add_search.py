import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tqdm import tqdm
import clip
import torch

def faiss_add(features_dim, image_features):
  index = faiss.IndexFlatL2(features_dim)
  print("The model is ", index.is_trained)
  print("Insert process\n")
  for feature_vector in tqdm(image_features):
    index.add(feature_vector)
  return index

def search(input_text, index, k):
    D, I = index.search(input_text,k)
    return D, I

def visual_result(I, image_dataset):
    num_images = len(I[0])
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Create subplots

    i = 0
    for index in I[0]:
        axes[i].imshow(image_dataset[index])
        axes[i].axis('off')  # Turn off axis labels
        i += 1
    plt.show()
