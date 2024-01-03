# Import necessary packages.
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.models as models

from tqdm.auto import tqdm
import os
import pandas as pd


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


# Data Augmentation
train_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.RandomCrop(224, 50, True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):

    def __init__(self, data_path, images_list, labels=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images_list = images_list
        self.labels = labels

    # return image(tensor) and label
    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = Image.open(os.path.join(self.data_path, image_path))
#         label = int(image_path.split("_")[0])
        label = -1
        if self.labels:
            label = int(self.labels[index])
            
        if self.transform:
            image = self.transform(image)
        
        return image, label

    # return the length of dataset
    def __len__(self):
        return len(self.images_list)


def model_training(train_data_path, valid_data_path, batch_size=8, epochs=50, learning_rate=0.0001):
    train_images = []
    train_labels = []
    for image in os.listdir(train_data_path):
        train_images.append(image)
        train_labels.append(int(image.split("_")[0]))
    
    valid_images = []
    valid_labels = []
    for image in os.listdir(valid_data_path):
        valid_images.append(image)
        valid_labels.append(int(image.split("_")[0]))
        
    train_dataset = MyDataset(train_data_path, train_images, train_labels, train_tfm)
    valid_dataset = MyDataset(valid_data_path, valid_images, valid_labels, valid_tfm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = models.resnet50(pretrained=True) # weights="DEFAULT"
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
#     model.maxpool = Identity()
#     model.fc = nn.Linear(model.fc.in_features, 50)
    model = models.efficientnet_v2_l(weights='DEFAULT')
    model.classifier = nn.Sequential(
                       nn.Dropout(p=0.4, inplace=True),
                       nn.Linear(in_features=1280, out_features=50, bias=True))
    
    model.to(device)
    modelName = f"efficientnet_v2_l_pretrained_adam_224.ckpt"
    
    path = f"efficientnet_v2_l_pretrained_adam_224.txt"
    f = open(path, 'w')
    f.write(f"{modelName}\n")
    f.write(f"Epochs={epochs}\n")
    f.write(f"batch_size={batch_size}\n")
    f.write(f"learning_rate={learning_rate}\n")
    f.write(f"image size=(224)\n")
#     f.write(f"milestones: MultiStepLR *0.1, [5, 10, 15, 20]\n")
    f.write(f"reduce kernel size, stride")
    f.close()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1)
    
    notImprove = 0
    min_loss = 1000.
    train_acc = 0
    valid_acc = 0

    train_loss_list = []
    train_acc_list = []

    valid_loss_list = []
    valid_acc_list = []

    for epoch in range(epochs):
        model.train()

        train_loss = []
        train_accs = []
        
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device).long())
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            
            # Compute the gradients for parameters.
            loss.backward()
            
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            
            # Update the parameters with computed gradients.
            optimizer.step()
            
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            
            # Record the loss and accuracy.
            train_loss.append(loss.cpu().item())
            train_accs.append(acc.cpu())
        
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
                
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device).long())

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.cpu().item())
            valid_accs.append(acc.cpu())

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        
        scheduler.step()

        if valid_loss < min_loss:
            # Save model if your model improved
            min_loss = valid_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f}, accuracy = {:.4f})'.format(epoch + 1, min_loss, valid_acc))
            torch.save(model.state_dict(), f"G:/DLCV_HW1/hw1_p1_result/{modelName}")  # Save model to specified path
            notImprove = 0
        else:
            notImprove = notImprove + 1

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        if (epoch+1 == 1) or (epoch+1 == 6):
            torch.save(model.state_dict(), f"G:/DLCV_HW1/hw1_p1_result/{epoch+1}_{modelName}")  # Save model at specified epoch
            
        if epoch == 10:
            notImprove = 0
        if notImprove >= 5 and epoch >= 10:
            break
            
    train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
    train_acc_arr = np.array(train_acc_list).reshape(1,len(train_loss_list))
    valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))
    valid_acc_arr = np.array(valid_acc_list).reshape(1,len(train_loss_list))

    training_log = np.append(train_loss_arr,train_acc_arr,axis=0)
    training_log = np.append(training_log,valid_loss_arr,axis=0)
    training_log = np.append(training_log,valid_acc_arr,axis=0)

    np.savetxt(f"G:/DLCV_HW1/hw1_p1_result/efficientnet_v2_l_pretrained_adam_224_train_log.csv", training_log, delimiter=",")


if __name__ == "__main__":
    train_set = f"G:/DLCV_HW1/hw1_data_4_students/hw1_data/p1_data/train_50"
    val_set = f"G:/DLCV_HW1/hw1_data_4_students/hw1_data/p1_data/val_50"
    model_training(train_set, val_set)

