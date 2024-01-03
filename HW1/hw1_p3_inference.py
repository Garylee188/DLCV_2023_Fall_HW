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
from torchvision import models
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


test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# vgg16 = models.vgg16(weights='DEFAULT')
# vgg16 = vgg16.features
# class VGG16_FCN32s(nn.Module):
#     def __init__(self, num_classes):
#         super(VGG16_FCN32s, self).__init__()
#         self.features = vgg16
#         self.classifier = nn.Sequential(
#             nn.Conv2d(512, 4096, kernel_size=1),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Conv2d(4096, 4096, kernel_size=1),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Conv2d(4096, num_classes, kernel_size=1),
#             nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
    
class MyDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    # return image(tensor) and label
    def __getitem__(self, index):
#         images_list = sorted(glob.glob(f"{self.data_path}/*.jpg"))
        images_list = os.listdir(self.data_path)
        image_path = images_list[index]
        image = Image.open(os.path.join(self.data_path, image_path))

        if self.transform:
            image = self.transform(image)

        return image

    # return the length of dataset
    def __len__(self):
        return len(os.listdir(self.data_path))
    

def mask_to_color(predictions):
    # predictions: numpy arr (512, 512)
    color_map = {0: [0, 255, 255],
                 1: [255, 255, 0],
                 2: [255, 0, 255],
                 3: [0, 255, 0],
                 4: [0, 0, 255],
                 5: [255, 255, 255],
                 6: [0, 0, 0]}
    
    height, width = predictions.shape
    color_mask = np.zeros((3, height, width))

    for key, value in color_map.items():
        position = (predictions==key)
        color_mask[0, position] = value[0]
        color_mask[1, position] = value[1]
        color_mask[2, position] = value[2]

    return color_mask
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images_dir", type=str, default=f'G:/DLCV_HW1/hw1_data_4_students/hw1_data/p3_data/validation', 
                        help="testing images directory")
    parser.add_argument("--output_images_dir", type=str, default=f'./p3_model/1_pred_dir', help="output images directory")
    parser.add_argument("--model_path", type=str, default=f'./hw1_model_inference/14_fcn_resnet50_new.ckpt', help="Path of model")
    args = parser.parse_args()
    
    test_images_dir = args.test_images_dir
    output_images_dir = args.output_images_dir
    model_path = args.model_path
    
    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir)

    test_img_list = os.listdir(test_images_dir)
    test_dataset = MyDataset(test_images_dir, test_tfm)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model = VGG16_FCN32s(num_classes=7)
    
    model = models.segmentation.fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 7, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
    
#     model_path = "./p3_model"
#     model_name = "fcn_resnet50.ckpt"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    model.eval()
    
    predictions = []
    groundtruths = []
    
    # Iterate the validation set by batches.
    for i, batch in enumerate(tqdm(test_loader)):
        
        test_image_name = test_img_list[i]
        if test_image_name[-4:] == '.jpg':
            mask_image_name = test_image_name.split("_")[0] + "_mask.png"

            # A batch consists of image data and corresponding labels.
            imgs = batch

            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
            pred = torch.argmax(logits['out'], dim=1)

            np_pred = pred.cpu().detach().numpy()
            np_pred = np_pred.reshape((np_pred.shape[1], np_pred.shape[2]))

            mask = mask_to_color(np_pred)
            mask_image = Image.fromarray(mask.astype(np.uint8).transpose(1, 2, 0))

            mask_image.save(os.path.join(output_images_dir, mask_image_name))

if __name__ == "__main__":
    main()
