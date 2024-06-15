# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # MIT License

    # Copyright (c) 2021 apchenstu

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers, seed_everything
from pytorch_lightning.loggers import WandbLogger

import os
import time
import numpy as np
import imageio
import lpips
from skimage.metrics import structural_similarity as ssim

from model.geo_reasoner import CasMVSNet
from model.self_attn_renderer import Renderer
from utils.rendering import render_rays
from utils.utils import (
    load_ckpt,
    init_log,
    get_rays_pts,
    SL1Loss,
    SL1Loss_UniMatch,
    self_supervision_loss,
    self_supervision_loss_for_unimatch,
    img2mse,
    mse2psnr,
    acc_threshold,
    abs_error,
    visualize_depth,
)
from utils.options import config_parser
from data.get_datasets import (
    get_training_dataset,
    get_validation_dataset,
)

lpips_fn = lpips.LPIPS(net="vgg")

class GeoNeRF(LightningModule):
    def __init__(self, hparams):
        super(GeoNeRF, self).__init__()
        self.hparams.update(vars(hparams))
        self.wr_cntr = 0

        self.depth_loss = SL1Loss()
        self.depth_loss_unimatch = SL1Loss_UniMatch()
        self.learning_rate = hparams.lrate

        # Create geometry_reasoner and renderer models
        self.geo_reasoner = CasMVSNet(use_depth=False).cuda()
        self.renderer = Renderer(
            nb_samples_per_ray=hparams.nb_coarse + hparams.nb_fine
        ).cuda()
        self.scaler = torch.cuda.amp.GradScaler()

        self.eval_metric = [0.01, 0.05, 0.1]

        self.automatic_optimization = False
        self.save_hyperparameters()

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        # to unnormalize image for visualization
        device = data.device
        mean = (
            torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
            .view(*shape)
            .to(device)
        )
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def prepare_data(self):
        ## Generalizable
        self.train_dataset, self.train_sampler = get_training_dataset(self.hparams)
        self.val_dataset = get_validation_dataset(self.hparams)
        
    def configure_optimizers(self):
        eps = 1e-5

        self.renderer_learning_rate = self.learning_rate
        self.casmvs_learning_rate = self.learning_rate
        self.unimatch_learning_rate = self.learning_rate * 0.1

        self.renderer_params = list(self.renderer.parameters())
        self.casmvs_params = list(self.geo_reasoner.feature.parameters()) + \
                            list(self.geo_reasoner.cost_reg_0.parameters()) + \
                            list(self.geo_reasoner.cost_reg_1.parameters()) + \
                            list(self.geo_reasoner.cost_reg_2.parameters())
        self.unimatch_params = list(self.geo_reasoner.feature_flow_attn.parameters()) + \
                            list(self.geo_reasoner.refine_proj.parameters()) + \
                            list(self.geo_reasoner.refine.parameters())
        self.all_params = self.renderer_params + self.casmvs_params + self.unimatch_params

        opt = torch.optim.Adam(
            [{'params': self.renderer_params, 'lr': self.renderer_learning_rate,'name': "renderer"},
            {'params': self.casmvs_params, 'lr': self.casmvs_learning_rate, 'name': "casmvs"},
            {'params': self.unimatch_params, 'lr': self.unimatch_learning_rate, 'name': "unimatch"}],
            lr=self.learning_rate,
            #eps=1e-4,
            betas=(0.9, 0.999),
        )
        sch = CosineAnnealingLR(opt, T_max=self.hparams.num_steps, eta_min=eps)

        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            num_workers=8,
            batch_size=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=1,
            batch_size=1,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        loss = 0
        nb_views = self.hparams.nb_views
        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        ## Inferring Geometry Reasoner
        self.geo_reasoner.backbone.eval()
        self.geo_reasoner.transformer.eval()
        for p in self.geo_reasoner.backbone.parameters():
	        p.requires_grad = False
        for p in self.geo_reasoner.transformer.parameters():
	        p.requires_grad = False

        with torch.cuda.amp.autocast():
            feats_vol, feats_fpn, depth_map, depth_map_stereo, depth_values = self.geo_reasoner(
                imgs=batch["images"][:, :nb_views],
                imgs_aug=batch["images_aug"][:, :nb_views],
                coef=batch["coef"][:, :nb_views],
                affine_mats=batch["affine_mats"][:, :nb_views],
                affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
                near_far=batch["near_fars"][:, :nb_views],
                closest_idxs=batch["closest_idxs"][:, :nb_views],
                gt_depths=batch["depths_aug"][:, :nb_views],
            )

            ## Normalizing depth maps in NDC coordinate
            depth_map_norm = {}
            for l in range(3):
                depth_map_norm[f"level_{l}"] = (
                    depth_map[f"level_{l}"].detach() - depth_values[f"level_{l}"][:, :, 0]
                ) / (
                    depth_values[f"level_{l}"][:, :, -1]
                    - depth_values[f"level_{l}"][:, :, 0]
                )

            unpre_imgs = self.unpreprocess(batch["images"])

            (
                pts_depth,
                rays_pts,
                rays_pts_ndc,
                rays_dir,
                rays_gt_rgb,
                rays_gt_inv_depth,
                rays_pixs,
            ) = get_rays_pts(
                H,
                W,
                batch["c2ws"],
                batch["w2cs"],
                batch["intrinsics"],
                batch["near_fars"],
                depth_values,
                self.hparams.nb_coarse,
                self.hparams.nb_fine,
                nb_views=nb_views,
                train=True,
                train_batch_size=self.hparams.batch_size,
                target_img=unpre_imgs[0, -1],
                target_depth=batch["inv_depths_fd"][0, -1],
            )

            ## Rendering
            rendered_rgb, rendered_depth, loss_mask = render_rays(
                c2ws=batch["c2ws"][0, :nb_views],
                rays_pts=rays_pts,
                rays_pts_ndc=rays_pts_ndc,
                pts_depth=pts_depth,
                rays_dir=rays_dir,
                feats_vol=feats_vol,
                feats_fpn=feats_fpn[:, :nb_views],
                imgs=unpre_imgs[:, :nb_views],
                depth_map_norm=depth_map_norm,
                renderer_net=self.renderer,
            )

            # Supervising depth maps with either ground truth depth or self-supervision loss
            ## This loss is only used in the generalizable model
            ## if ground truth is available
            if isinstance(batch["inv_depths_fd_h"], dict):
                inv_depth_map = {}
                inv_depth_map_unimatch = []
                for l in range(3):
                    inv_depth_map[f"level_{l}"] = 1. / depth_map[f"level_{l}"]
                for l in range(len(depth_map_stereo)):
                    inv_depth_map_unimatch.append(1. / depth_map_stereo[l])
                loss = loss + 0.1 * self.depth_loss(inv_depth_map, batch["inv_depths_fd_h"], batch["near_fars"])
                loss = loss + 0.1 * self.depth_loss_unimatch(inv_depth_map_unimatch, batch["inv_depths_fd_h"], batch["near_fars"])
        
                loss = loss + 0.1 * self_supervision_loss(
                    self.depth_loss,
                    rays_pixs,
                    rendered_depth.detach(),
                    depth_map,
                    rays_gt_rgb,
                    unpre_imgs,
                    rendered_rgb.detach(),
                    batch["intrinsics"],
                    batch["c2ws"],
                    batch["w2cs"],
                    loss_mask,
                    for_unimatch_mask=batch["depths_fd"].detach(),
                    near_fars=batch["near_fars"]
                )
                loss = loss + 0.1 *  self_supervision_loss_for_unimatch(
                    self.depth_loss_unimatch,
                    rays_pixs,
                    rendered_depth.detach(),
                    depth_map_stereo,
                    rays_gt_rgb,
                    unpre_imgs,
                    rendered_rgb.detach(),
                    batch["intrinsics"],
                    batch["c2ws"],
                    batch["w2cs"],
                    loss_mask,
                    for_unimatch_mask=batch["depths_fd"].detach(),
                    near_fars=batch["near_fars"]
                )
                if loss != 0:
                    self.log("train/dlosspgt", loss.item(), prog_bar=False)

            ## Supervising ray depths
            depth_available = True
            if depth_available:
                inv_rendered_depth = 1. / rendered_depth
                rend_dloss = 0.1 * self.depth_loss(inv_rendered_depth, rays_gt_inv_depth, batch["near_fars"][0, -1])
                loss = loss + rend_dloss
                if rend_dloss != 0:
                    self.log("train/rend_dlossgt", rend_dloss.item(), prog_bar=False)

            ## Reconstruction loss
            mse_loss = img2mse(rendered_rgb, rays_gt_rgb, loss_mask)
            loss = loss + mse_loss

            with torch.no_grad():
                self.log("train/loss", loss.item(), prog_bar=True)
                psnr = mse2psnr(mse_loss.detach())
                self.log("train/PSNR", psnr.item(), prog_bar=False)
                self.log("train/img_mse_loss", mse_loss.item(), prog_bar=False)

        opt = self.optimizers()
        sch = self.lr_schedulers()

        # Warming up the learning rate
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )  
            assert opt.param_groups[0]["name"] == "renderer"
            assert opt.param_groups[1]["name"] == "casmvs"
            assert opt.param_groups[2]["name"] == "unimatch"
            opt.param_groups[0]["lr"] = lr_scale * self.renderer_learning_rate
            opt.param_groups[1]["lr"] = lr_scale * self.casmvs_learning_rate
            opt.param_groups[2]["lr"] = lr_scale * self.unimatch_learning_rate

        self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False)

        # Manual Optimization
        scaled_loss = self.scaler.scale(loss)
        self.manual_backward(scaled_loss)
        self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(self.unimatch_params, 1.0)
        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad()
        sch.step()

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        ## This makes Batchnorm to behave like InstanceNorm
        self.geo_reasoner.train()

        log_keys = [
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "val_depth_loss_r",
            "val_abs_err",
            "mask_sum",
        ] + [f"val_acc_{i}mm" for i in self.eval_metric]
        log = {}
        log = init_log(log, log_keys)

        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        nb_views = self.hparams.nb_views

        self.geo_reasoner.backbone.eval()
        self.geo_reasoner.transformer.eval()
        self.geo_reasoner.feature_flow_attn.eval()
        self.geo_reasoner.refine_proj.eval()
        self.geo_reasoner.refine.eval()
        for p in self.geo_reasoner.backbone.parameters():
	        p.requires_grad = False
        for p in self.geo_reasoner.transformer.parameters():
	        p.requires_grad = False
        for p in self.geo_reasoner.feature_flow_attn.parameters():
	        p.requires_grad = False
        for p in self.geo_reasoner.refine_proj.parameters():
	        p.requires_grad = False
        for p in self.geo_reasoner.refine.parameters():
	        p.requires_grad = False

        with torch.no_grad():
            ## Inferring Geometry Reasoner
            feats_vol, feats_fpn, depth_map, depth_map_stereo, depth_values = self.geo_reasoner(
                imgs=batch["images"][:, :nb_views],
                coef=batch["coef"][:, :nb_views],
                affine_mats=batch["affine_mats"][:, :nb_views],
                affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
                near_far=batch["near_fars"][:, :nb_views],
                closest_idxs=batch["closest_idxs"][:, :nb_views],
                gt_depths=batch["depths_aug"][:, :nb_views] if self.hparams.eval_dataset_name == "real" else batch["depths"][:, :nb_views],
            )

            ## Normalizing depth maps in NDC coordinate
            depth_map_norm = {}
            for l in range(3):
                depth_map_norm[f"level_{l}"] = (
                    depth_map[f"level_{l}"] - depth_values[f"level_{l}"][:, :, 0]
                ) / (
                    depth_values[f"level_{l}"][:, :, -1]
                    - depth_values[f"level_{l}"][:, :, 0]
                )

            unpre_imgs = self.unpreprocess(batch["images"])

            rendered_rgb, rendered_depth = [], []
            for chunk_idx in range(
                H * W // self.hparams.chunk + int(H * W % self.hparams.chunk > 0)
            ):
                pts_depth, rays_pts, rays_pts_ndc, rays_dir, _, _, _ = get_rays_pts(
                    H,
                    W,
                    batch["c2ws"],
                    batch["w2cs"],
                    batch["intrinsics"],
                    batch["near_fars"],
                    depth_values,
                    self.hparams.nb_coarse,
                    self.hparams.nb_fine,
                    nb_views=nb_views,
                    chunk=self.hparams.chunk,
                    chunk_idx=chunk_idx,
                )

                ## Rendering
                rend_rgb, rend_depth, loss_mask = render_rays(
                    c2ws=batch["c2ws"][0, :nb_views],
                    rays_pts=rays_pts,
                    rays_pts_ndc=rays_pts_ndc,
                    pts_depth=pts_depth,
                    rays_dir=rays_dir,
                    feats_vol=feats_vol,
                    feats_fpn=feats_fpn[:, :nb_views],
                    imgs=unpre_imgs[:, :nb_views],
                    depth_map_norm=depth_map_norm,
                    renderer_net=self.renderer,
                )
                rendered_rgb.append(rend_rgb)
                rendered_depth.append(rend_depth)
            rendered_rgb = torch.clamp(
                torch.cat(rendered_rgb).reshape(H, W, 3).permute(2, 0, 1), 0, 1
            )
            rendered_depth = torch.cat(rendered_depth).reshape(H, W)

            ## Check if there is any ground truth depth information for the dataset
            depth_available = (batch["depths_h"].sum() > 0) if self.hparams.eval_dataset_name == "real" else (batch["depths"].sum() > 0)

            ## Evaluate only on pixels with meaningful ground truth depths
            if depth_available:
                mask = (batch["depths_h"] > 0) if self.hparams.eval_dataset_name == "real" else (batch["depths"] > 0)
                img_gt_masked = (unpre_imgs[0, -1] * mask[0, -1][None]).cpu()
                rendered_rgb_masked = (rendered_rgb * mask[0, -1][None]).cpu()
            else:
                img_gt_masked = unpre_imgs[0, -1].cpu()
                rendered_rgb_masked = rendered_rgb.cpu()

            unpre_imgs = unpre_imgs.cpu()
            rendered_rgb, rendered_depth = rendered_rgb.cpu(), rendered_depth.cpu()
            img_err_abs = (rendered_rgb_masked - img_gt_masked).abs()

            if self.hparams.eval_dataset_name:
                depth_target = batch["depths_h"][0, -1].cpu()
                mask_target = depth_target > 0
            else:
                depth_target = batch["depths"][0, -1].cpu()
                mask_target = (depth_target > 0) * (depth_target < 10e2) * (rendered_depth > 0)

            if depth_available:
                log["val_psnr"] = mse2psnr(torch.mean(img_err_abs[:, mask_target] ** 2))
            else:
                log["val_psnr"] = mse2psnr(torch.mean(img_err_abs**2))
            log["val_ssim"] = ssim(
                rendered_rgb_masked.permute(1, 2, 0).numpy(),
                img_gt_masked.permute(1, 2, 0).numpy(),
                data_range=1,
                multichannel=True,
            )
            log["val_lpips"] = lpips_fn(
                rendered_rgb_masked[None] * 2 - 1, img_gt_masked[None] * 2 - 1
            ).item()  # Normalize to [-1,1]

            depth_minmax = [
                0.9 * batch["near_fars"].min().detach().cpu().numpy(),
                1.1 * batch["near_fars"].max().detach().cpu().numpy(),
            ]
            rendered_depth_vis, _ = visualize_depth(rendered_depth, depth_minmax)

            if depth_available:
                log["val_abs_err"] = abs_error(
                    rendered_depth, depth_target, mask_target
                ).sum()
                log[f"val_acc_{self.eval_metric[0]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[0]
                ).sum()
                log[f"val_acc_{self.eval_metric[1]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[1]
                ).sum()
                log[f"val_acc_{self.eval_metric[2]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[2]
                ).sum()
                log["mask_sum"] = mask_target.float().sum()

            img_vis = (
                torch.cat(
                    (
                        unpre_imgs[:, -1],
                        torch.stack([rendered_rgb, img_err_abs * 5]),
                        rendered_depth_vis[None],
                    ),
                    dim=0,
                )
                .clip(0, 1)
                .permute(2, 0, 3, 1)
                .reshape(H, -1, 3)
                .numpy()
            )

            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.eval_dataset_name}/{self.hparams.expname}/rendered_results/",
                exist_ok=True,
            )
            imageio.imwrite(
                f"{self.hparams.logdir}/{self.hparams.eval_dataset_name}/{self.hparams.expname}/rendered_results/{self.wr_cntr:03d}.png",
                (
                    rendered_rgb.detach().permute(1, 2, 0).clip(0.0, 1.0).cpu().numpy()
                    * 255
                ).astype("uint8"),
            )

            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.eval_dataset_name}/{self.hparams.expname}/evaluation/",
                exist_ok=True,
            )
            imageio.imwrite(
                f"{self.hparams.logdir}/{self.hparams.eval_dataset_name}/{self.hparams.expname}/evaluation/{self.global_step:08d}_{self.wr_cntr:02d}.png",
                (img_vis * 255).astype("uint8"),
            )

            print(f"Image {self.wr_cntr:02d} rendered.")
            self.wr_cntr += 1

        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()
        mean_ssim = np.stack([x["val_ssim"] for x in outputs]).mean()
        mean_lpips = np.stack([x["val_lpips"] for x in outputs]).mean()
        mask_sum = torch.stack([x["mask_sum"] for x in outputs]).sum()
        mean_d_loss_r = torch.stack([x["val_depth_loss_r"] for x in outputs]).mean()
        mean_abs_err = torch.stack([x["val_abs_err"] for x in outputs]).sum() / mask_sum
        mean_acc_1mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[0]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_2mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[1]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_4mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[2]}mm"] for x in outputs]).sum()
            / mask_sum
        )

        self.log("val/PSNR", mean_psnr, prog_bar=False)
        self.log("val/SSIM", mean_ssim, prog_bar=False)
        self.log("val/LPIPS", mean_lpips, prog_bar=False)
        if mask_sum > 0:
            self.log("val/d_loss_r", mean_d_loss_r, prog_bar=False)
            self.log("val/abs_err", mean_abs_err, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[0]}mm", mean_acc_1mm, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[1]}mm", mean_acc_2mm, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[2]}mm", mean_acc_4mm, prog_bar=False)

        with open(
            f"{self.hparams.logdir}/{self.hparams.eval_dataset_name}/{self.hparams.expname}/{self.hparams.expname}_metrics.txt",
            "w",
        ) as metric_file:
            metric_file.write(f"PSNR: {mean_psnr}\n")
            metric_file.write(f"SSIM: {mean_ssim}\n")
            metric_file.write(f"LPIPS: {mean_lpips}")

        return


def load_model_unimatch(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        p = model.state_dict()[key]
        if key in state['model']:
            ip = state['model'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
                print('load layer: {}, in checkpoint'.format(key))
    return model


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    geonerf = GeoNeRF(args)

    ## Checking to logdir to see if there is any checkpoint file to continue with
    ckpt_path = f"{args.logdir}/{args.eval_dataset_name}/{args.expname}/ckpts"
    if os.path.isdir(ckpt_path) and len(os.listdir(ckpt_path)) > 0:
        ckpt_file = os.path.join(ckpt_path, os.listdir(ckpt_path)[-1])
    else:
        ckpt_file = None

    ## Setting a callback to automatically save checkpoints
    checkpoint_callback = ModelCheckpoint(
        f"{args.logdir}/{args.eval_dataset_name}/{args.expname}/ckpts",
        filename="ckpt_step-{step:06d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
    )

    ## Setting up a logger
    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.expname,
            project="GeoNeRF",
            save_dir=f"{args.logdir}",
            resume="allow",
            id=args.expname,
        )
    elif args.logger == "tensorboard":
        logger = loggers.TestTubeLogger(
            save_dir=f"{args.logdir}/{args.eval_dataset_name}/{args.expname}",
            name=args.expname + "_logs",
            debug=False,
            create_git_tag=False,
        )
    else:
        logger = None

    seed_everything(42, workers=True)
    args.use_amp = False if args.eval else True
    trainer = Trainer(
        max_steps=args.num_steps,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        resume_from_checkpoint=ckpt_file,
        logger=logger,
        progress_bar_refresh_rate=1,
        gpus=1,
        num_sanity_val_steps=0,
        val_check_interval=2000,
        deterministic=True,
        precision=32
    )

    if not args.eval:  ## Train
        ## Loading the pretrained weights from Cascade MVSNet & UniMatch
        ckpt_file = "pretrained_weights/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth"
        load_model_unimatch(geonerf.geo_reasoner, ckpt_file)

        torch.utils.model_zoo.load_url(
            "https://github.com/kwea123/CasMVSNet_pl/releases/download/1.5/epoch.15.ckpt",
            model_dir="pretrained_weights",
        )
        ckpt_file = "pretrained_weights/epoch.15.ckpt"
        load_ckpt(geonerf.geo_reasoner, ckpt_file, "model", strict=False)

        trainer.fit(geonerf)
    else:  ## Eval
        geonerf = GeoNeRF(args)

        ckpt_file = "ckpt_step-249999.ckpt"
        load_ckpt(geonerf.geo_reasoner, ckpt_file, "geo_reasoner")
        load_ckpt(geonerf.renderer, ckpt_file, "renderer")

        trainer.validate(geonerf)
