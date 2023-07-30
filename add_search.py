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

def faiss_add(feature_dim, image_features):
    index = faiss.IndexFlatL2(feature_dim)

    if not index.is_trained:
        print("Model Faiss chưa được train")
    if index.is_trained:
        print("\n\nStore vector process\n")
        for features_vector in tqdm(image_features):
            index.add(features_vector)
    return index

def search(text_feature, index, k):
    D, I = index.search(text_feature, k)
    return D, I

def visual_result(I, image_dataset):
    num_images = len(I[0])
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i, index in enumerate(I[0]):
        axes[i].imshow(image_dataset[index])
        axes[i].axis("off")
    plt.show()
