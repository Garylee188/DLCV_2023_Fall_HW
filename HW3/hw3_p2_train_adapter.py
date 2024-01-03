import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm
from PIL import Image
import os
import numpy as np
import json
from tqdm.auto import tqdm
# from decoder import Decoder, Config
from decoder_adapter import Decoder, Config
# from decoder_ptuning import Decoder, Config
from tokenizer import BPETokenizer
import loralib as lora


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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

class ImgCapDataset(Dataset):
    
    def __init__(self, img_dir, json_file, transform):
        self.img_dir = img_dir
        self.json_file = json_file
        self.transform = transform
        
        with open(self.json_file, 'r') as file:
            self.data_dict = json.load(file)
        
    def __getitem__(self, index):
        
        max_token_len = 300
        caption = self.data_dict['annotations'][index]['caption']
        img_id = self.data_dict['annotations'][index]['image_id']
        
        for img_info in self.data_dict['images']:
            img_name = img_info['file_name']
            if img_id == img_info['id']:
                img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
                encoding = BPETokenizer('./encoder.json', './vocab.bpe')
                token_for_model = token_for_gt = encoding.encode(caption)
                token_for_model = [50256] + token_for_model
                token_for_gt = token_for_gt + [50256]
                token_for_model = token_for_model + [50256] * (max_token_len-len(token_for_model))
                token_for_gt = token_for_gt + [-100] * (max_token_len-len(token_for_gt))
                
                return self.transform(img), torch.tensor(token_for_model), torch.tensor(token_for_gt)

    def __len__(self):
        return len(self.data_dict['annotations'])

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

def train_eval(model, train_loader, valid_loader, test_dir, json_file_path, n_epochs, lr, save_dir, device):
    
    # Lora: This sets requires_grad to False for all parameters without the string "lora_" in their names
    # lora.mark_only_lora_as_trainable(model)

    
    for param in model.parameters():
        param.requires_grad = False

    # Baseline, no PEFT
    # for param in model.decoder.parameters():
    #     param.requires_grad = True

    # Adapter: only update params on adapter and layer_norm which in decoder
    trained_layer = ['adapter', 'cross_attn', 'ln_1', 'ln_2', 'ln_3']
    for name, p in model.decoder.transformer.h.named_parameters():
        for l in trained_layer:
            if l in name:
                p.requires_grad = True
    
    # cross_att_trained_layer = ['cross_attn.img_attn', 'cross_attn.c_proj']
    # for name, p in model.decoder.transformer.h.named_parameters():
    #     for l in cross_att_trained_layer:
    #         if l in name:
    #             p.requires_grad = True

    for param in model.linear.parameters():
        param.requires_grad = True

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_param)
    if total_param >= 35000000:
        print('Too much params!')
        return

    trained_params = [name for name, param in model.named_parameters() if param.requires_grad == True]
    print(trained_params)

    # Start Training
    model.to(device)
    model.train()
    modelName = 'vl_model_huge_adapter.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
    minLoss = 1000.

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}')

        print('[Train]')
        train_loss = 0.0
        train_bar = tqdm(train_loader)
        for batch in train_bar:
            img, token_for_model, token_for_gt = batch
            
            logits = model(img.to(device), token_for_model.to(device))
            # logits : (batch, max_seq, vocab_num) = (8, 1024, 50257)
            # logits.permute(0, 2, 1) -> (8, 50257, 1024)
            # token_for_gt : (batch, max_seq) = (8, 1024)
            loss = criterion(logits.permute(0, 2, 1).to(device), token_for_gt.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_description(f"train loss: {loss.item():.4f}")
            train_loss += loss.item()
        
        trainLoss = train_loss / len(train_loader)
        print(f"Train Loss = {trainLoss:.5f}")
        
        print('[Validation]')
        model.eval()
        valid_loss = 0.0
        valid_bar = tqdm(valid_loader)
        for batch in valid_bar:
            img, token_for_model, token_for_gt = batch
            
            with torch.no_grad():
                logits = model(img.to(device), token_for_model.to(device))
                loss = criterion(logits.permute(0, 2, 1).to(device), token_for_gt.to(device))
            valid_loss += loss.item()
            valid_bar.set_description(f"valid loss: {loss.item():.4f}")

        validLoss = valid_loss / len(valid_loader)
        print(f"Validation Loss = {validLoss:.5f}")

        save_weights = {k: v for k, v in model.state_dict().items() if k in trained_params}
        torch.save(save_weights, os.path.join(save_dir, str((epoch+1)) + '_' + modelName))

        if validLoss < minLoss:
            minLoss = validLoss

            # Generate caption
            generate(model, test_dir, json_file_path, device)

            print('Saving Best model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, minLoss))
            torch.save(save_weights, os.path.join(save_dir, 'best_' + modelName))

def generate(model, test_dir, json_file_path, device):
    
    test_data = ImgDataset(test_dir, valid_tfm)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    model.to(device)
    model.eval()
    
    encoding = BPETokenizer('./encoder.json', './vocab.bpe')

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


if __name__ == '__main__':
    
    decoder_ckpt = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\decoder_model.bin'
    train_dir = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\images\train'
    valid_dir = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\images\val'
    test_dir = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\images\test'
    train_json = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\train.json'
    valid_json = r'C:\Users\ipmc_msi\Desktop\DLCV-HW3\hw3_data\p2_data\val.json'
    save_dir = r'E:\DLCV-HW3\p2'
    json_file_path = r'E:\DLCV-HW3\p2\best_pred_adapter.json'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = 4
    lr = 0.00008
    n_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare Dataset
    train_data = ImgCapDataset(train_dir, train_json, train_tfm)
    valid_data = ImgCapDataset(valid_dir, valid_json, valid_tfm)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    decoder_cfg = Config(checkpoint=decoder_ckpt)
    vl_model = VLModel(cfg=decoder_cfg, device=device)

    train_eval(
        model = vl_model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        test_dir = test_dir,
        json_file_path = json_file_path,
        n_epochs = n_epochs,
        lr = lr,
        save_dir = save_dir,
        device = device
    )

    state_dict = torch.load(r'E:\DLCV-HW3\p2\best_vl_model_huge_adapter.pth')
    vl_model.load_state_dict(state_dict, strict=False)
    generate(
        model = vl_model,
        test_dir = valid_dir, 
        json_file_path = json_file_path,
        device = device
    )
  