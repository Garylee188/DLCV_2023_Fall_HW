import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
# from datasets import dataset_dict
from dataset import *

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        # self.save_hyperparameters()
        self.hp = hparams
        self.loss = loss_dict[hparams.loss_type]()
        self.val_step_loss = []
        self.train_step_loss = []
        self.val_step_psnr = []
        self.train_step_psnr = []

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hp.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hp.chunk],
                            self.hp.N_samples,
                            self.hp.use_disp,
                            self.hp.perturb,
                            self.hp.noise_std,
                            self.hp.N_importance,
                            self.hp.chunk, # chunk size is effective in val mode
                            False)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        # dataset = dataset_dict[self.hparams.dataset_name]
        # kwargs = {'root_dir': self.hparams.root_dir,
        #           'img_wh': tuple(self.hparams.img_wh)}
        # if self.hparams.dataset_name == 'llff':
        #     kwargs['spheric_poses'] = self.hparams.spheric_poses
        #     kwargs['val_num'] = self.hparams.num_gpus
        # self.train_dataset = dataset(split='train', **kwargs)
        # self.val_dataset = dataset(split='val', **kwargs)

        self.train_dataset = KlevrDataset(self.hp.root_dir, split='train', get_rgb=True)
        self.val_dataset = KlevrDataset(self.hp.root_dir, split='val', get_rgb=True)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hp, self.models)
        scheduler = get_scheduler(self.hp, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hp.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=1, num_workers=4)
    
    def training_step(self, batch, batch_idx):
        # log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        train_loss = self.loss(results, rgbs)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            train_psnr = psnr(results[f'rgb_{typ}'], rgbs)
            # log['train/psnr'] = psnr_
            self.log('train_psnr', train_psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.train_step_loss.append(train_loss)
        self.train_step_psnr.append(train_psnr)

        return train_loss
        # return {'loss': loss,
        #         'progress_bar': {'train_psnr': psnr_},
        #         'log': log
        #        }

    def on_train_epoch_end(self):
        mean_loss = torch.stack(self.train_step_loss).mean()
        mean_psnr = torch.stack(self.train_step_psnr).mean()
        self.train_step_loss.clear()
        self.train_step_psnr.clear()
        print(f"Train | Loss={mean_loss}, PSNR={mean_psnr}")

    def validation_step(self, batch, batch_idx):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        # log = {'val_loss': self.loss(results, rgbs)}
        val_loss = self.loss(results, rgbs)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_idx == 0:
            W, H = self.hp.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        # log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        val_psnr = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('val_psnr', val_psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_step_loss.append(val_loss)
        self.val_step_psnr.append(val_psnr)
        return val_loss

    def on_validation_epoch_end(self):
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        # return {'progress_bar': {'val_loss': mean_loss,
        #                          'val_psnr': mean_psnr},
        #         'log': {'val/loss': mean_loss,
        #                 'val/psnr': mean_psnr}
        #        }
        # val_loss = self.trainer.callback_metrics['val_loss'].item()

        mean_loss = torch.stack(self.val_step_loss).mean()
        mean_psnr = torch.stack(self.val_step_psnr).mean()
        self.val_step_loss.clear()
        self.val_step_psnr.clear()
        print(f"Validation | Loss={mean_loss}, PSNR={mean_psnr}")


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath='ckpts/exp5/',
                                          filename='{epoch}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_top_k=3)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
        # debug=False,
        # create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=checkpoint_callback,
                    #   resume_from_checkpoint='./ckpts/exp5/epoch=2.ckpt',
                      logger=logger,
                      # early_stop_callback=None,
                      # weights_summary=None,
                      # progress_bar_refresh_rate=1,
                      enable_progress_bar=True,
                      # gpus=hparams.num_gpus,
                      # distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                    #   profiler=hparams.num_gpus==1
                    )

    trainer.fit(system, ckpt_path='./ckpts/exp5/epoch=2.ckpt')