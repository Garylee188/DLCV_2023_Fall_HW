import os
import glob
import numpy as np
import random
from tqdm.auto import tqdm
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import DatasetFolder


def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
set_seed(0)


# Data Augmentation
train_tfm = transforms.Compose([
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def mask_to_label(self, index):
        color_map = {0: [0, 255, 255],
                     1: [255, 255, 0],
                     2: [255, 0, 255],
                     3: [0, 255, 0],
                     4: [0, 0, 255],
                     5: [255, 255, 255],
                     6: [0, 0, 0]}

        mask_imgs = sorted(glob.glob(f"{self.data_path}/*.png"))
        rgb_img = np.array(Image.open(mask_imgs[index]))

        height, width, channels = rgb_img.shape
        label = np.zeros((7, height, width))

        for key, value in color_map.items():
            label[key, :, :] = np.all(rgb_img==np.array(value), axis=2)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return label_tensor

    # return image(tensor) and label
    def __getitem__(self, index):
        images_list = sorted(glob.glob(f"{self.data_path}/*.jpg"))
        image = Image.open(images_list[index])
        label = self.mask_to_label(index)

        if self.transform:
            image = self.transform(image)

        return image, label

    # return the length of dataset
    def __len__(self):
        return len(glob.glob(f"{self.data_path}/*.jpg"))


vgg16 = models.vgg16(weights='DEFAULT')
vgg16 = vgg16.features

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN32s, self).__init__()
        self.features = vgg16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


def model_training(train_data_dir, valid_data_dir, save_dir, num_epochs, batch_size, learning_rate):

    train_dataset = MyDataset(train_data_dir, train_tfm)
    valid_dataset = MyDataset(valid_data_dir, valid_tfm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== model A: vgg16 + FCN32s ==== #
    # model = VGG16_FCN32s(num_classes=7)
    # modelName = f"vgg16-fcn32.ckpt"

    # ==== model B: resnet101 + FCN ==== #
    model = models.segmentation.fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 7, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    modelName = f"fcn_resnet50_new.ckpt"

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = 1000.
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(num_epochs):

        model.train()
        train_loss = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            labels = labels.float()

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            # loss = criterion(logits, labels.to(device))  # vgg-16
            loss = criterion(logits['out'], labels.to(device))  # resnet-50

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Record the loss and accuracy.
            train_loss.append(loss.cpu().item())

        train_loss = sum(train_loss) / len(train_loss)
        train_loss_list.append(train_loss)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        predictions = []
        groundtruths = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            labels = labels.float()
            mask = torch.argmax(labels, dim=1)
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            pred = torch.argmax(logits, dim=1)  # vgg-16
            pred = torch.argmax(logits['out'], dim=1)  # resnet-50

            # We can still compute the loss (but not the gradient).
            # loss = criterion(logits, labels.to(device))  # vgg-16
            loss = criterion(logits['out'], labels.to(device))  # resnet-50

            # Record the loss and accuracy.
            valid_loss.append(loss.cpu().item())

            groundtruths.append(mask.cpu())
            predictions.append(pred.cpu())

        groundtruths = np.concatenate(groundtruths, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        mean_iou = mean_iou_score(predictions, groundtruths)
        print(f"mean IOU = ", mean_iou)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_loss_list.append(valid_loss)

        if valid_loss < min_loss:
            # Save model if your model improved
            min_loss = valid_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_loss,))
            torch.save(model.state_dict(), f"{save_dir}/best_{modelName}")  # Save model to specified path

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

        torch.save(model.state_dict(), f"{save_dir}/{epoch+1}_{modelName}")  # Save model at specified epoch

    train_loss_arr = np.array(train_loss_list).reshape(1,len(train_loss_list))
    valid_loss_arr = np.array(valid_loss_list).reshape(1,len(train_loss_list))

    training_log = np.append(train_loss_arr, valid_loss_arr,axis=0)
    np.savetxt(f"{save_dir}/{modelName[:-5]}_train_log.csv", training_log, delimiter=",")


if __name__ == "__main__":
    model_training(f"/content/drive/MyDrive/Colab Notebooks/hw1_data_4_students/hw1_data/p3_data/train",
                   f"/content/drive/MyDrive/Colab Notebooks/hw1_data_4_students/hw1_data/p3_data/validation",
                   f"/content/drive/MyDrive/Colab Notebooks/hw1_p3_result", 50, 4, 0.0001)
