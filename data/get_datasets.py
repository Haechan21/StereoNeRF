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

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np

from data.llff_stereo import LLFF_Stereo_Dataset
from data.front_stereo import FRONT_Stereo_Dataset


def get_training_dataset(args, downsample=1.0):
    train_datasets = [
        LLFF_Stereo_Dataset(
            root_dir=args.real_train_root_path,
            disp_dir=args.real_train_disp_path,
            split="train",
            max_len=-1,
            img_wh=(512,256),
            nb_views=args.nb_views,
            imgs_folder_name="images",
        )
    ] * 9   # Simple Repeat. It is just for convinient for validation
    weights = [1.0] * 9 # Simple Repeat. It is just for convinient for validation

    train_weights_samples = []
    for dataset, weight in zip(train_datasets, weights):
        num_samples = len(dataset)
        weight_each_sample = weight / num_samples
        train_weights_samples.extend([weight_each_sample] * num_samples)

    train_dataset = ConcatDataset(train_datasets)
    train_weights = torch.from_numpy(np.array(train_weights_samples))
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    return train_dataset, train_sampler


def get_validation_dataset(args, downsample=1.0):
    if not args.eval:
        max_len = 2
    else:
        max_len = -1

    if args.eval_dataset_name == "real":
        val_dataset = LLFF_Stereo_Dataset(
            root_dir=args.real_val_root_path,
            disp_dir=args.real_val_disp_path,
            split="val",
            max_len=max_len,
            img_wh=(512,256),
            nb_views=args.nb_views,
            imgs_folder_name="images",
        )
    elif args.eval_dataset_name == "synthetic":
        val_dataset = FRONT_Stereo_Dataset(
            root_dir=args.synthetic_root_path,
            depth_root_dir=args.synthetic_depth_path,
            split="val",
            max_len=max_len,
            img_wh=(512,256),
            nb_views=args.nb_views,
            imgs_folder_name="images",
        )
        

    return val_dataset
