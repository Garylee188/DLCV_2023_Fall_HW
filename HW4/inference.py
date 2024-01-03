import torch
from utils import *
from collections import defaultdict
import argparse

from models.rendering import *
from models.nerf import *

from dataset import *
import os
import cv2
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

img_wh = (256, 256)

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)

nerf_coarse = NeRF()
nerf_fine = NeRF()

ckpt_path = './best_nerf.ckpt'

load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

nerf_coarse.cuda().eval()
nerf_fine.cuda().eval()

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]

N_samples = 64
N_importance = 128
use_disp = False
chunk = 1024*32 # (1024, 256=64+192) 128+128+256=512

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default='./dataset')
    parser.add_argument("--output_dir", type=str, default='./pred_result')
    args = parser.parse_args()
    
    meta_dir = args.meta_dir
    output_dir = args.output_dir
    split = 'test'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_dataset = KlevrDataset(meta_dir, split, get_rgb=False)
    with open(os.path.join(meta_dir, "metadata.json"), "r") as file:
        meta = json.load(file)
    split_ids = meta['split_ids'][split]

    for idx in tqdm(range(test_dataset.__len__())):
        rays = test_dataset[idx]['rays'].cuda()
        # image_id = test_dataset[idx]['img_idx']
        image_id = split_ids[idx]
        results = f(rays)
        torch.cuda.synchronize()
        img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
        cv2.imwrite(os.path.join(output_dir, f'{image_id:05d}.png'), img_pred * 255)

