import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
import torchvision.transforms as transforms
import random
import os
import csv
import numpy as np
import pandas as pd
import argparse
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)


train_tfm = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.Grayscale(1),   
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    transforms.RandomCrop((28, 28), padding=5),
    transforms.ToTensor(),
])

valid_tfm = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

usps_valid_tfm = transforms.Compose([
    transforms.ToTensor(),
])


class Digit_Dataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None):
        self.data_path = data_path
        self.csv_file = csv_file
        self.transform = transform
        
        df = pd.read_csv(self.csv_file)
        self.image_name = df['image_name']
        self.label = df['label']
        
    # return image(tensor) and label
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_name[index])
        image = Image.open(image_path)
        label = self.label[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    # return the length of dataset
    def __len__(self):
        return len(self.image_name)


def get_data(batch_size=32):
    
    digit_path = r'C:\Users\ipmc_msi\Desktop\DLCV-HW2\2023_hw2_data\hw2_data\digits'

    mnistm_data_path = os.path.join(digit_path, 'mnistm\data')
    svhn_data_path = os.path.join(digit_path, 'svhn\data')
    usps_data_path = os.path.join(digit_path, 'usps\data')

    mnistm_train_csv = os.path.join(digit_path, 'mnistm', 'train.csv')
    mnistm_valid_csv = os.path.join(digit_path, 'mnistm', 'val.csv')
    svhn_train_csv = os.path.join(digit_path, 'svhn', 'train.csv')
    svhn_valid_csv = os.path.join(digit_path, 'svhn', 'val.csv')
    usps_train_csv = os.path.join(digit_path, 'usps', 'train.csv')
    usps_valid_csv = os.path.join(digit_path, 'usps', 'val.csv')

    mnistm_train_set = Digit_Dataset(mnistm_data_path, mnistm_train_csv, train_tfm)
    mnistm_valid_set = Digit_Dataset(mnistm_data_path, mnistm_valid_csv, valid_tfm)
    mnistm_train_loader = DataLoader(mnistm_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    mnistm_valid_loader = DataLoader(mnistm_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    svhn_train_set = Digit_Dataset(svhn_data_path, svhn_train_csv, train_tfm)
    svhn_valid_set = Digit_Dataset(svhn_data_path, svhn_valid_csv, valid_tfm)
    svhn_train_loader = DataLoader(svhn_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    svhn_valid_loader = DataLoader(svhn_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    usps_train_set = Digit_Dataset(usps_data_path, usps_train_csv, usps_valid_tfm)
    usps_valid_set = Digit_Dataset(usps_data_path, usps_valid_csv, usps_valid_tfm)
    usps_train_loader = DataLoader(usps_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    usps_valid_loader = DataLoader(usps_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return {
        "mnistm_train_loader": mnistm_train_loader,
        "mnistm_valid_loader": mnistm_valid_loader,
        "svhn_train_loader": svhn_train_loader,
        "svhn_valid_loader": svhn_valid_loader,
        "usps_train_loader": usps_train_loader,
        "usps_valid_loader": usps_valid_loader,
        "mnistm_valid_set": mnistm_valid_set,
        "usps_valid_set": usps_valid_set
    }


class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=1, out_channel=32):
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
    def __init__(self, in_channel=64, out_channel=256, in_dim=7, labels_num=10):
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
    def __init__(self, in_channel=64, out_channel=256, in_dim=7):
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


def train_eval(feature_extractor, label_predictor, domain_classifier,
               src_train_loader, src_valid_loader, tgt_train_loader, tgt_valid_loader,
               n_epochs, lr, batch_size, device, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()
    
    feature_opt = torch.optim.Adam(feature_extractor.parameters(), lr=lr)
    label_opt = torch.optim.Adam(label_predictor.parameters(), lr=lr)
    domain_opt = torch.optim.Adam(domain_classifier.parameters(), lr=lr)
    
    feature_scheduler = torch.optim.lr_scheduler.MultiStepLR(feature_opt, milestones=[5,10,15], gamma=0.1)
    label_scheduler = torch.optim.lr_scheduler.MultiStepLR(label_opt, milestones=[5,10,15], gamma=0.1)
    domain_scheduler = torch.optim.lr_scheduler.MultiStepLR(domain_opt, milestones=[5,10,15], gamma=0.1)
    
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()  # binary
    
    gt_domain = torch.cat((torch.ones(batch_size, 1), torch.zeros(batch_size, 1)), 0).to(device)  # 1 for src, 0 for tgt
    min_loss = 10000.
    
    for epoch in range(n_epochs):
        print(f'Epoch = {epoch+1}')
        print('[Train]')
        train_loss = []
        label_loss_list, domain_loss_list = [], []
        for batch in tqdm(src_train_loader):
            src_imgs, src_labels = batch
            tgt_sample = iter(tgt_train_loader)
            tgt_imgs, _ = tgt_sample.next()  # tgt domain only use images
            
            src_imgs, src_labels, tgt_imgs = src_imgs.to(device), src_labels.to(device), tgt_imgs.to(device)
            
            # Train Domain Classifier
            feature = feature_extractor(torch.cat([src_imgs, tgt_imgs], 0))
            pred_domain = domain_classifier(feature.detach())
            domain_loss = domain_criterion(pred_domain, gt_domain)
            
            domain_opt.zero_grad()
            
            domain_loss.backward()
            
            domain_opt.step()
            
            # Train Label Predictor
            src_logits = label_predictor(feature[:batch_size])
            pred_domain = domain_classifier(feature)
            
            label_loss = label_criterion(src_logits, src_labels)
            domain_loss = domain_criterion(pred_domain, gt_domain)
            
            adapt_lambda = 0.31  # from paper
            total_loss = label_loss - adapt_lambda * domain_loss
            
            feature_opt.zero_grad()
            label_opt.zero_grad()
            domain_opt.zero_grad()
            
            total_loss.backward()
            
            feature_opt.step()
            label_opt.step()
            
            label_loss_list.append(label_loss.cpu().item())
            domain_loss_list.append(domain_loss.cpu().item())
            train_loss.append(total_loss.cpu().item())
        
        feature_scheduler.step()
        label_scheduler.step()
        domain_scheduler.step()
        
        train_avg_loss = sum(train_loss) / len(train_loss)
        print(f"Train Loss = {train_avg_loss:.5f}")
        
        # Evaluation (for tgt validation set)
        valid_avg_loss, valid_avg_acc = evaluation(
            feature_extractor = feature_extractor,
            label_predictor = label_predictor,
            label_criterion = label_criterion,
            valid_loader = tgt_valid_loader,
            device = device
        )
        
        print(f"Valid Loss = {valid_avg_loss:.5f}, Valid Accuracy = {valid_avg_acc:.5f}")
        
        
        if (epoch+1) % 10 == 0:
            F_model_name = str(epoch+1) + '_Feature_Extractor.ckpt'
            L_model_name = str(epoch+1) + '_Label_Predictor.ckpt'
            D_model_name = str(epoch+1) + '_Domain_Classifier.ckpt'
            torch.save(feature_extractor.state_dict(), os.path.join(save_dir, F_model_name))
            torch.save(label_predictor.state_dict(), os.path.join(save_dir, L_model_name))
            torch.save(domain_classifier.state_dict(), os.path.join(save_dir, D_model_name))
        
        if valid_avg_loss < min_loss:
            min_loss = valid_avg_loss
            torch.save(feature_extractor.state_dict(), os.path.join(save_dir, 'Best_Feature_Extractor.ckpt'))
            torch.save(label_predictor.state_dict(), os.path.join(save_dir, 'Best_Label_Predictor.ckpt'))
            torch.save(domain_classifier.state_dict(), os.path.join(save_dir, 'Best_Domain_Classifier.ckpt'))
        

def evaluation(feature_extractor, label_predictor, label_criterion, valid_loader, device):
    
    feature_extractor.eval()
    label_predictor.eval()
    valid_acc, valid_loss = [], []
    
    print('[Evaluation]')
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            logits = label_predictor(feature_extractor(imgs))
            loss = label_criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            valid_loss.append(loss.cpu().item())
            valid_acc.append(acc.cpu())

    valid_avg_loss = sum(valid_loss) / len(valid_loss)
    valid_avg_acc = sum(valid_acc) / len(valid_acc)
    
    return valid_avg_loss, valid_avg_acc


def visualization(feature_extractor, src_valid_data, tgt_valid_data, save_dir, device):
    
    feature_extractor.eval()
    src_data_len = src_valid_data.__len__()
    tgt_data_len = tgt_valid_data.__len__()
    all_domain_labels = list(np.ones(src_data_len)) + list(np.zeros(tgt_data_len))
    
    all_valid_data = ConcatDataset([src_valid_data, tgt_valid_data])
    all_valid_loader = DataLoader(all_valid_data, batch_size=1, shuffle=False, num_workers=0)
    
    print('[TSNE]')
    features = []
    labels = []
    for batch in tqdm(all_valid_loader):
        with torch.no_grad():
            img, label = batch
            labels.extend(label.tolist())
            feature = feature_extractor(img.to(device))
            features.extend(feature.cpu().numpy())
            
    features = np.array(features).reshape(len(all_valid_loader.dataset), -1)
    
    tsne = TSNE(n_components=2).fit_transform(features)
    
    # if plot domain, labels -> all_domain_labels
    cmap = plt.get_cmap('tab20b')
    colors = plt.cm.get_cmap('tab20b', len(set(labels)))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne[:, 0], tsne[:, 1], s=10, c=labels, cmap=colors)

    plt.title('t-SNE by class')  # class or domain
    plt.savefig(f'{save_dir}/tsne_class.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = r"E:\DLCV-HW2\p3_result\target_usps_3"
    
    lr = 0.0001
    batch_size = 128
    n_epochs = 20
    
    data = get_data(batch_size=batch_size)
    
    F = FeatureExtractor().to(device)
    L = LabelPredictor().to(device)
    D = DomainClassifier().to(device)
    
    train_eval(
        feature_extractor = F,
        label_predictor = L,
        domain_classifier = D,
        src_train_loader = data["mnistm_train_loader"],
        src_valid_loader = data["mnistm_valid_loader"],
        tgt_train_loader = data["usps_train_loader"],
        tgt_valid_loader = data["usps_valid_loader"],
        n_epochs = n_epochs,
        lr = lr,
        batch_size = batch_size,
        device = device,
        save_dir = save_dir
    )
    
    F.load_state_dict(torch.load(r'E:\DLCV-HW2\p3_result\target_usps_3\20_Feature_Extractor.ckpt'))
    L.load_state_dict(torch.load(r'E:\DLCV-HW2\p3_result\target_usps_3\20_Label_Predictor.ckpt'))
    valid_loss, valid_acc = evaluation(
        feature_extractor = F,
        label_predictor = L,
        label_criterion = nn.CrossEntropyLoss(),
        valid_loader = data["usps_valid_loader"],
        device = device
    )
    print(valid_acc)

    visualization(
        feature_extractor = F,
        src_valid_data = data["mnistm_valid_set"],
        tgt_valid_data = data["usps_valid_set"],
        save_dir = save_dir,
        device = device
    )
