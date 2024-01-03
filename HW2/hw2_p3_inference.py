import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import random
import os
import csv
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm.auto import tqdm


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)

valid_tfm = transforms.Compose([
    transforms.ToTensor(),
])


class DigitDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    # return image(tensor)
    def __getitem__(self, index):
        images_list = sorted(os.listdir(self.data_path))
        image_path = images_list[index]
        image = Image.open(os.path.join(self.data_path, image_path))
            
        if self.transform:
            image = self.transform(image)
        
        return image

    # return the length of dataset
    def __len__(self):
        return len(os.listdir(self.data_path))


    
class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=3, out_channel=32):
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel*2, 3, 1, 1),
            nn.BatchNorm2d(out_channel*2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, 3, 1, 1),
            nn.BatchNorm2d(out_channel*2),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)  # (out_channel, 28, 28)
        x = self.maxpool(self.conv2(x))  # (out_channel, 14, 14)
        x = self.maxpool(self.conv3(x))  # (out_channel*2, 7, 7)
        x = self.conv4(x)  # (out_channel*2, 7, 7)
        x = x.view(x.size(0), -1)
        
        return x


class LabelPredictor(nn.Module):
    def __init__(self, in_channel=64, out_channel=512, in_dim=7, labels_num=10):
        super(LabelPredictor, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_channel*in_dim*in_dim, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel//2),
            nn.ReLU(),
            nn.Linear(out_channel//2, labels_num),
        )
    
    def forward(self, x):
        x = self.fc(x)
        
        return x
    
        
class DomainClassifier(nn.Module):
    def __init__(self, in_channel=64, out_channel=128, in_dim=7):
        super(DomainClassifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_channel*in_dim*in_dim, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel//2),
            nn.ReLU(),
            nn.Linear(out_channel//2, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc(x))

        return x
    

def write_csv(output_path, predictions, data_path):
    ''' write csv file of image_name and label '''
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'label'])
        for i, label in enumerate(predictions):
            image_name = sorted(os.listdir(data_path))[i]
            writer.writerow([image_name, str(label)])
    
    
def inference(feature_extractor, label_predictor, test_dir, csv_file, device):
    
    test_set = DigitDataset(test_dir, valid_tfm)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
    
    feature_extractor.eval()
    label_predictor.eval()
    
    predictions = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            imgs = batch
            imgs = imgs.to(device)
            
            logits = label_predictor(feature_extractor(imgs))
            pred = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(pred)
            
    write_csv(csv_file, predictions, test_dir)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='./2023_hw2_data/hw2_data/digits/svhn/data')
    parser.add_argument('--csv_file', type=str, default='./p3_result/test_pred.csv')
    args = parser.parse_args()
    
    test_dir = args.test_dir
    csv_file = args.csv_file
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if 'svhn' in test_dir:
        F = FeatureExtractor(in_channel=3).to(device)
        L = LabelPredictor(out_channel=512).to(device)
        D = DomainClassifier(out_channel=128).to(device)
        
        F.load_state_dict(torch.load('./svhn_feature_extractor.ckpt'))
        L.load_state_dict(torch.load('./svhn_label_predictor.ckpt'))
        D.load_state_dict(torch.load('./svhn_domain_classifier.ckpt'))
        
        inference(feature_extractor=F, label_predictor=L, test_dir=test_dir, csv_file=csv_file, device=device)
        
    if 'usps' in test_dir:
        F = FeatureExtractor(in_channel=1).to(device)
        L = LabelPredictor(out_channel=256).to(device)
        D = DomainClassifier(out_channel=256).to(device)
        
        F.load_state_dict(torch.load('./usps_feature_extractor.ckpt'))
        L.load_state_dict(torch.load('./usps_label_predictor.ckpt'))
        D.load_state_dict(torch.load('./usps_domain_classifier.ckpt'))
        
        inference(feature_extractor=F, label_predictor=L, test_dir=test_dir, csv_file=csv_file, device=device)