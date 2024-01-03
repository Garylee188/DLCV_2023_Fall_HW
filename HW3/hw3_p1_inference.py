import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import clip
from PIL import Image
import os
import numpy as np
import json
import csv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse

def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

set_seed(0)

class MyDataset(Dataset):
    def __init__(self, images_dir, transform):
        self.images_dir = images_dir
        self.transform = transform
    
    def __getitem__(self, index):
        images_list = os.listdir(self.images_dir)
        image_name = images_list[index]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_name
    
    def __len__(self):
        return len(os.listdir(self.images_dir))

def write_csv(output_path, predictions, data_path):
    ''' write csv file of filenames and predicted labels '''
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i, label in enumerate(predictions):
            filename = os.listdir(data_path)[i]
            writer.writerow([filename, str(label)]) 

def inference(model, preprocess, test_images_dir, data_dict, csv_file_path, device):
    dataset = MyDataset(test_images_dir, preprocess)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data_dict.values()]).to(device)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, images_name = batch
            images = images.to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(images, text)
            preds = torch.argmax(logits_per_image.cpu(), dim=-1).tolist()
            predictions.extend(preds)
    write_csv(csv_file_path, predictions, test_images_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images_dir", type=str, default='./hw3_data/p1_data/val')
    parser.add_argument("--json_file_path", type=str, default='./hw3_data/p1_data/id2label.json')
    parser.add_argument("--csv_file_path", type=str, default='./hw3_data/p1_data/pred.csv')
    args = parser.parse_args()
    
    test_images_dir = args.test_images_dir
    json_file_path = args.json_file_path
    csv_file_path = args.csv_file_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    with open(json_file_path, 'r') as file:
        data_dict = json.load(file)
    inference(
        model = model,
        preprocess = preprocess,
        test_images_dir = test_images_dir,
        data_dict = data_dict,
        csv_file_path = csv_file_path,
        device = device
    )
    
if __name__ == '__main__':
    main()
    
