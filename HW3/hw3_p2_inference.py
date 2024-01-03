import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm
from PIL import Image
import os
import numpy as np
import json
from tqdm.auto import tqdm
from decoder import Decoder, Config
from tokenizer import BPETokenizer
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

# Data Augmentation
valid_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImgDataset(Dataset):
    
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        
    def __getitem__(self, index):
        img_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, img_list[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_name = img_list[index].split('.')[0]
        return img, img_name

    def __len__(self):
        return len(os.listdir(self.img_dir))

class VLModel(nn.Module):
    
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.vit = timm.create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True, num_classes=0).to(device)
        self.linear = nn.Linear(1280, 768)
        self.decoder = Decoder(cfg).to(device)
        
    def forward(self, image, caption):
        x_enc = self.vit.forward_features(image)
        x_enc = self.linear(x_enc)
        res = self.decoder(caption, x_enc)
        return res

def generate(model, test_dir, json_file_path, device):

    encoding = BPETokenizer('./encoder.json', './vocab.bpe')
    test_data = ImgDataset(test_dir, valid_tfm)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()

    prediction = {}
    # Start generating caption
    for batch in tqdm(test_loader):
        img, img_name = batch
        img_name = img_name[0]
        
        with torch.no_grad():
            pred_token = [50256]
            while len(pred_token) <= 1 or pred_token[-1] != 50256:
                if len(pred_token) == 70:
                    break
                logits = model(img.to(device), torch.tensor(pred_token).unsqueeze(0).to(device))
                prob = F.softmax(logits[:, -1, :], dim=-1)
                token = torch.argmax(prob, dim=-1).view(-1).tolist()[-1]
                pred_token.append(int(token))
            
            pred_caption = encoding.decode(pred_token)
            pred_caption = pred_caption.replace('<|endoftext|>', '')
            prediction[str(img_name)] = pred_caption
    
    with open(json_file_path, 'w') as json_file:
        json.dump(prediction, json_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images_dir", type=str, default='./hw3_data/p2_data/images/val')
    parser.add_argument("--json_file_path", type=str, default='./pred.json')
    parser.add_argument("--decoder_weights", type=str, default='./hw3_data/p2_data/decoder_model.bin')
    args = parser.parse_args()
    
    test_images_dir = args.test_images_dir
    json_file_path = args.json_file_path
    decoder_weights = args.decoder_weights

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder_cfg = Config(checkpoint=decoder_weights)
    vl_model = VLModel(cfg=decoder_cfg, device=device)
    vl_model.load_state_dict(torch.load('./best_vl_model_huge_lora.pth'), strict=False)
    generate(
        model = vl_model,
        test_dir = test_images_dir, 
        json_file_path = json_file_path,
        device = device
    )

if __name__ == '__main__':
    main()
  