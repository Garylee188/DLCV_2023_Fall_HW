import torch
import os
from tqdm.auto import tqdm
from PIL import Image
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


images_path = f"G:/DLCV_HW1/hw1_data_4_students/hw1_data/p2_data/mini/train"
all_images = []
for image in tqdm(os.listdir(images_path)):
    img = Image.open(os.path.join(images_path, image))
    tensor_img = transform(img)
    all_images.append(tensor_img)
all_images = torch.stack(all_images, dim=0)
train_loader = DataLoader(all_images, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet50()
resnet.to(device)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool',
    projection_size = 256,           # the projection size
    projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
    moving_average_decay = 0.99      # the moving average decay factor for the target encoder, already set at what paper recommends
)

train_loader = DataLoader(all_images, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
opt = torch.optim.Adam(learner.parameters(), lr=0.0001)

for i in range(100):
    print(f"Epoch {i+1}")
    loss_record = []
    for batch in tqdm(train_loader):
        loss = learner(batch.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        loss_record.append(loss)
    train_loss = sum(loss_record) / len(loss_record)
    print(f"Train Loss : {train_loss}")
    torch.save(resnet.state_dict(), f'C:/Users/ipmc_msi/Desktop/DLCV_HW1/hw1_data_4_students/hw1_data/p2_data/{i+1}_byol_ssl_backbone.pt')

# save your improved network
torch.save(resnet.state_dict(), f'C:/Users/ipmc_msi/Desktop/DLCV_HW1/hw1_data_4_students/hw1_data/p2_data/best_byol_ssl_backbone.pt')
