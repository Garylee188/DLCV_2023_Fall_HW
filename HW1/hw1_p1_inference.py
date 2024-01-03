# Import necessary packages.
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.models as models

from tqdm import tqdm
import os
import pandas as pd
import argparse

class MyDataset(Dataset):
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

##### For Inference #####
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

##### Data Processing #####
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images_dir", type=str, default='./hw1_data_4_students/hw1_data/p1_data/val_50', 
                        help="Path to the test data directory")
    parser.add_argument("--csv_file_path", type=str, default='./hw1_p1_result/pred.csv', help="Path of output csv file")
    parser.add_argument("--model_path", type=str, default='./hw1_model_inference/hw1_p1_modelB.ckpt', help="Path of model")
    args = parser.parse_args()

    test_images_dir = args.test_images_dir
    csv_file_path = args.csv_file_path
    model_path = args.model_path

    test_dataset = MyDataset(test_images_dir, test_tfm)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

#     model_path = 'G:/DLCV_HW1/hw1_p1_result'
#     model_name = 'efficientnet_v2_l_pretrained_adam_224.ckpt'
    model = models.efficientnet_v2_l(weights='DEFAULT')
    model.classifier = nn.Sequential(
                       nn.Dropout(p=0.4, inplace=True),
                       nn.Linear(in_features=1280, out_features=50, bias=True))
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    predictions = []
 
    # Iterate the validation set by batches.
    for batch in tqdm(test_loader):

        # A batch consists of image data.
        imgs = batch
        
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            pred = model(imgs.to(device))
        prediction = pred.argmax(dim=-1).cpu().tolist()
        predictions.extend(prediction)
        
    write_csv(csv_file_path, predictions, test_images_dir)

if __name__ == "__main__":
    main()
