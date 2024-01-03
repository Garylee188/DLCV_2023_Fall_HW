import torch
import torchvision.transforms as T
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.rendering import *
from models.nerf import *

from dataset import *
import os
import cv2
from PIL import Image
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

img_wh = (256, 256)

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)

nerf_coarse = NeRF()
nerf_fine = NeRF()

ckpt_path = './ckpts/exp5/epoch=14.ckpt'

load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

nerf_coarse.cuda().eval()
nerf_fine.cuda().eval()

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]

N_samples = 128
N_importance = 256
use_disp = False
chunk = 1024*32*16 # (1024, 256=64+192) 128+128+256=512

@torch.no_grad()
def f(rays):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        False,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def make_grid_plt(images, save_path):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1, 1])

    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        ax = plt.subplot(gs[i])
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    # plt.close()

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    x_ = cv2.cvtColor(x_, cv2.COLOR_RGB2BGR)
    # x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    # x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

json_dir = './dataset'
split = 'val'   # change to 'test'
pred_dir = './result/depth2'

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

test_dataset = KlevrDataset(json_dir, split, get_rgb=False)
depth_images = []
for idx in tqdm(range(test_dataset.__len__())):
    rays = test_dataset[idx]['rays'].cuda()
    image_id = test_dataset[idx]['img_idx']
    results = f(rays)
    torch.cuda.synchronize()
    depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])
    depth_pred = visualize_depth(depth_pred)
    
    # plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
    # plt.axis("off")
    # plt.savefig(os.path.join(pred_dir, f'{image_id:05d}_depth.png'), bbox_inches="tight")
    cv2.imwrite(os.path.join(pred_dir, f'{image_id:05d}_depth.png'), depth_pred)
    depth_images.append(os.path.join(pred_dir, f'{image_id:05d}_depth.png'))
make_grid_plt(depth_images, os.path.join(pred_dir, 'depth_grid.png'))

# print('PSNR', metrics.psnr(img_gt, img_pred).item())

# plt.subplots(figsize=(15, 8))
# plt.tight_layout()
# plt.subplot(221)
# plt.title('GT')
# plt.imshow(img_gt)
# plt.subplot(222)
# plt.title('pred')
# plt.imshow(img_pred)
# plt.subplot(223)
# plt.title('depth')
# plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
# plt.show()

