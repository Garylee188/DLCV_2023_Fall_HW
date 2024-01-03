# Import necessary packages.
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset,  DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.models as models

from tqdm import tqdm
import os
import pandas as pd
import argparse

class MyDataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        self.data_path = data_path
        self.csv_file = csv_file
        self.transform = transform
        
        df = pd.read_csv(self.csv_file)
        self.image_ids = df['id']
        self.image_names = df['filename']

    # return image(tensor)
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_names[index])
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image

    # return the length of dataset
    def __len__(self):
        return len(self.image_names)
    

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
def write_csv(output_path, predictions, test_loader):
    ''' write csv file of filenames and predicted labels '''
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'filename', 'label'])
        for i, label in enumerate(predictions):
            image_id = test_loader.dataset.image_ids[i]
            filename = test_loader.dataset.image_names[i]
            writer.writerow([str(image_id), filename, str(label)])

##### Data Processing #####
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_path", type=str, default='./hw1_data_4_students/hw1_data/p2_data/office/val.csv', 
                        help="Path to the images csv file")
    parser.add_argument("--test_images_dir", type=str, default='./hw1_data_4_students/hw1_data/p2_data/office/val', 
                        help="Path to the folder containing images")
    parser.add_argument("--csv_file_path", type=str, default='./p2_model/test_pred.csv', help="Path of output csv file")
    parser.add_argument("--model_path", type=str, default='./hw1_model_inference/Setting_C_new.ckpt', help="Path of model")
    args = parser.parse_args()
    
    test_csv_path = args.test_csv_path
    test_images_dir = args.test_images_dir
    csv_file_path = args.csv_file_path
    model_path = args.model_path
    
    test_dataset = MyDataset(test_images_dir, test_csv_path, test_tfm)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    backbone = models.resnet50()

    # Self-defined MLP
    classifier = nn.Sequential(
                    nn.Linear(backbone.fc.out_features, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 65)
                )
    
    # Complete Model
    model = nn.Sequential(
                backbone,
                classifier
            )
    
#     model_path = './p2_model'
#     model_name = 'Setting_C.ckpt'
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
        
    write_csv(csv_file_path, predictions, test_loader)

if __name__ == "__main__":
    main()
