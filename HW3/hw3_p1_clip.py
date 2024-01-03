import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import clip
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, images_dir, transform):
        self.images_dir = images_dir
        self.transform = transform
        
    # return image(tensor) and label
    def __getitem__(self, index):
        images_list = os.listdir(self.images_dir)
        image_path = os.path.join(self.images_dir, images_list[index])
        image = Image.open(image_path)
        image = self.transform(image)
        label = int(images_list[index].split('_')[0])
#         text = data_dict[label]
        
        return image, label

    # return the length of dataset
    def __len__(self):
        return len(os.listdir(self.images_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def cal_accuracy(data_loader, data_dict):
    accuracy = []
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data_dict.values()]).to(device)
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, labels = batch
            images = images.to(device)
            image_features = model.encode_image(images)
            # print(image_features.size())
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(images, text)
#             probs = logits_per_image.softmax(dim=-1).cpu()
            preds = torch.argmax(logits_per_image.cpu(), dim=-1)
            accuracy.append((preds == labels).float().mean().item())
    
    acc = sum(accuracy) / len(accuracy)
    return acc

def top5_scores_visulization(image, data_dict):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data_dict.values()]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = similarity[0].topk(5)
    top_probs = top_probs.cpu().numpy() * 100
    top_labels = top_labels.cpu().numpy()
    
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(2, 2 ,2)
    y = np.arange(top_probs.shape[-1])
    # y = np.arange(0, 101, 20)
#     plt.grid()
    plt.xlim(0, 100)
    plt.barh(y, top_probs)
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [f"a photo of a {data_dict[str(index)]}" for index in top_labels])
#     plt.xlabel("probability (%)")

    plt.subplots_adjust(wspace=0.5)
    plt.show()

if __name__ == '__main__':
    dir_path = r'C:\Users\yychen\Desktop\DLCV_HW3\hw3_data\p1_data'
    
    with open(os.path.join(dir_path, 'id2label.json'), 'r') as file:
        data_dict = json.load(file)
    
    dataset = MyDataset(os.path.join(dir_path, 'val'), preprocess)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    acc = cal_accuracy(data_loader, data_dict)
    
    img_list = os.listdir(os.path.join(dir_path, 'val'))
    img = random.sample(img_list, 1)
    image = Image.open(os.path.join(dir_path, 'val', img[0]))
    top5_scores_visulization(image, data_dict)

