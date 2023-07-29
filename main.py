from preprocessing import load_data, image_db
from add_search import faiss_add, search, visual_result
import torch
import clip



def encode_text(input_text, model):
  tokenized_text = torch.cat([clip.tokenize(input_text)]).to(device)
  with torch.no_grad():
    text_feature = model.encode_text(tokenized_text)
  text_feature /= text_feature.norm(dim=-1, keepdim=True)
  return text_feature


def main():
  image_dataset = load_data("/content/images")
  image_features,  model = image_db(image_dataset)
  index = faiss.IndexFlatL2(image_features.shape[1], image_features)
  text = input()
  text_encode = encode_text(text, model)
  D, I = search(text_encode, image_features, 10)
  visual_result(I, image_dataset)

  
