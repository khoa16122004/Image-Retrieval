from preprocessing import load_data, image_db
from add_search import faiss_add, search, visual_result
import torch
import clip
import faiss


def encode_text(input_text, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_text = torch.cat([clip.tokenize(input_text)]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(tokenized_text)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    return text_feature

def main():
    image_dataset = load_data(r"D:\AIC\animals_dataset")
    model, image_features = image_db(image_dataset)
    index = faiss_add(image_features[0].shape[1], image_features)
    print(type(index))
    text = input("Your querry: ")
    text_encode = encode_text(text, model)
    D, I = search(text_encode, index, 10)
    visual_result(I, image_dataset)

main()
