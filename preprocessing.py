import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tqdm import tqdm
import clip
import torch


def load_data(image_folder):
  image_dataset = []
  for file_name in tqdm(os.listdir(image_folder)):
    path = os.path.join(image_folder, file_name)
    image_dataset.append(Image.open(path))
  print(f"\n\nDataset has {len(image_dataset)} image")
  return image_dataset

def image_db(image_dataset):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load('ViT-B/32', device) # [['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']]

  image_feature = [] # features list
  with torch.no_grad():
    print("\n\nEncode processing: ")
    for image in tqdm(image_dataset):
      image_feature.append( model.encode_image(preprocess(image).unsqueeze(0).to(device)))

  image_features_normalized = [feature / torch.norm(feature, dim=-1, keepdim=True) for feature in image_feature] # chuẩn hóa

  return image_features_normalized
