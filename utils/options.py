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

import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="Config file path")

    # Datasets options
    parser.add_argument("--eval_dataset_name", type=str, default="llff", choices=["real", "synthetic"],)
    parser.add_argument("--real_train_root_path", type=str, help="Path to Real Train dataset")
    parser.add_argument("--real_train_disp_path", type=str, help="Path to Disp of Real Train dataset")
    parser.add_argument("--real_val_root_path", type=str, help="Path to Real Valid dataset")
    parser.add_argument("--real_val_disp_path", type=str, help="Path to Disp of Real Valid dataset")
    parser.add_argument("--synthetic_root_path", type=str, help="Path to Synthetic Valid dataset")
    parser.add_argument("--synthetic_depth_path", type=str, help="Path to Depth of Synthetic Valid dataset")

    # Training options
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--nb_views", type=int, default=3)
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Gradually warm-up learning rate in optimizer")

    # Rendering options
    parser.add_argument("--chunk", type=int, default=4096, help="Number of rays rendered in parallel")
    parser.add_argument("--nb_coarse", type=int, default=96, help="Number of coarse samples per ray")
    parser.add_argument("--nb_fine", type=int, default=32, help="Number of additional fine samples per ray",)

    # Other options
    parser.add_argument("--expname", type=str, help="Experiment name")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--logdir", type=str, default="./logs/", help="Where to store ckpts and logs")
    parser.add_argument("--eval", action="store_true", help="Render and evaluate the test set")

    return parser.parse_args()
