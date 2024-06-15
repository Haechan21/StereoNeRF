import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import cv2
import glob
import numpy as np
import pickle
from PIL import Image

from utils.utils import get_nearest_pose_ids, read_pfm


class FRONT_Stereo_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        depth_root_dir,
        split,
        nb_views,
        sparsity=0,
        img_wh=(512,256),
        max_len=-1,
        scene="None",
        imgs_folder_name="images",
        depths_folder_name="depths",
        poses_folder_name="poses",
    ):
        self.root_dir = root_dir
        self.depth_root_dir = depth_root_dir
        self.split = split
        self.nb_views = nb_views // 2
        self.sparsity = sparsity
        self.scene = scene
        self.imgs_folder_name = imgs_folder_name
        self.depths_folder_name = depths_folder_name
        self.poses_folder_name = poses_folder_name

        self.max_len = max_len
        self.img_wh = img_wh
        self.org_img_wh = [864, 448]
        self.downsample_ratio = [self.img_wh[0] / self.org_img_wh[0], self.img_wh[1] / self.org_img_wh[1]]

        self.define_transforms()
        self.define_bounds = np.array([[0.5, 8.0]])
        self.baseline = 0.08 # pre-defined in blenderproc
        tf_matrix = np.zeros((4, 4))
        np.fill_diagonal(tf_matrix, 1)
        tf_matrix[0, -1] = self.baseline
        self.tf_matrix = tf_matrix
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        self.build_metas()

    def define_transforms(self):
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_metas(self):
        if self.scene != "None":
            self.scans = [
                os.path.basename(scan_dir)
                for scan_dir in sorted(
                    glob.glob(os.path.join(self.root_dir, self.scene))
                )
            ]
        else:
            self.scans = [
                os.path.basename(scan_dir)
                for scan_dir in sorted(glob.glob(os.path.join(self.root_dir, "*")))
            ]

        self.meta = []
        self.poses = {}
        self.poses_other = {}
        self.image_paths = {}
        self.image_other_paths = {}
        self.depth_paths = {}
        self.depth_other_paths = {}
        self.pose_paths = {}
        self.pose_other_paths = {}
        self.near_far = {}
        self.id_list = {}
        self.closest_idxs = {}
        self.c2ws = {}
        self.w2cs = {}
        self.co2ws = {}
        self.w2cos = {}
        self.intrinsics = {}
        self.affine_mats = {}
        self.affine_mats_inv = {}
        self.affine_other_mats = {}
        self.affine_other_mats_inv = {}
        for scan in self.scans[::-1]:
            k_matrxix_path = os.path.join(self.root_dir, scan, self.poses_folder_name, "intrinsic.pkl")
            del_index_path = os.path.join(self.root_dir, scan, "del_index.txt")
    
            del_index = open(del_index_path, "r").readlines()
            del_index = list(map(lambda s: int(s.strip()), del_index))
            with open(k_matrxix_path, 'rb') as f:
                k_matrix = pickle.load(f)
                refine_k_matrix = k_matrix.copy()
                refine_k_matrix[0, :] = refine_k_matrix[0, :] * self.downsample_ratio[0]
                refine_k_matrix[1, :] = refine_k_matrix[1, :] * self.downsample_ratio[1]

            self.image_paths[scan] = sorted(
                glob.glob(os.path.join(self.root_dir, scan, self.imgs_folder_name, "left", "*.png"))
            )
            self.image_other_paths[scan] = sorted(
                glob.glob(os.path.join(self.root_dir, scan, self.imgs_folder_name, "right", "*.png"))
            )
            self.depth_paths[scan] = sorted(
                glob.glob(os.path.join(self.depth_root_dir, scan, self.depths_folder_name, "left", "*.pkl"))
            ) 
            self.depth_other_paths[scan] = sorted(
                glob.glob(os.path.join(self.depth_root_dir, scan, self.depths_folder_name, "right", "*.pkl"))
            )
            self.pose_paths[scan] = sorted(
                glob.glob(os.path.join(self.root_dir, scan, self.poses_folder_name, "left", "*.pkl"))
            )

            assert len(self.image_paths[scan]) == len(self.depth_paths[scan]), print("img2depth", scan)
            assert len(self.depth_paths[scan]) == len(self.depth_other_paths[scan]), print("depth2depth", scan)

            bounds = self.define_bounds.copy()
            bounds = np.repeat(bounds, len(self.image_paths[scan]), axis=0)  # (N_images, 2)
            
            poses, poses_other = [], []
            for p_path in self.pose_paths[scan]:
                with open(p_path, 'rb') as f:
                    p_pose = pickle.load(f)
                po_pose = np.matmul(p_pose, self.tf_matrix) 
                p_pose = p_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                po_pose = po_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                poses.append(p_pose[:3, :4])
                poses_other.append(po_pose[:3, :4])
            
            poses = np.stack(poses)
            poses_other = np.stack(poses_other)
            self.poses[scan] = poses.astype('float32')
            self.poses_other[scan] = poses_other.astype('float32')
            self.near_far[scan] = bounds.astype('float32')

            num_viewpoint = len(self.image_paths[scan])
            val_ids = [idx for idx in range(0, num_viewpoint, 10)]
            w, h = self.img_wh

            self.id_list[scan] = []
            self.closest_idxs[scan] = []
            self.c2ws[scan] = []
            self.w2cs[scan] = []
            self.co2ws[scan] = []
            self.w2cos[scan] = []
            self.intrinsics[scan] = []
            self.affine_mats[scan] = []
            self.affine_mats_inv[scan] = []
            self.affine_other_mats[scan] = []
            self.affine_other_mats_inv[scan] = []
            for idx in range(num_viewpoint):
                view_ids = get_nearest_pose_ids(
                    poses[idx, :, :],
                    ref_poses=poses[..., :],
                    num_select=self.nb_views + 1,
                    angular_dist_method="matrix",
                )

                if (
                    (self.split == "val" and idx in val_ids) 
                    or (
                        self.split == "train"
                        and self.scene != "None"
                    )
                    or (self.split == "train" and self.scene == "None")
                ):
                    if len(view_ids) == self.nb_views + 1:
                        self.meta.append({"scan": scan, "target_idx": idx})
                    else:
                        view_ids = get_nearest_pose_ids(
                            poses[idx, :, :],
                            ref_poses=poses[..., :],
                            num_select=self.nb_views + 1,
                            angular_dist_method="matrix",
                        ) # dummy data
                    
                self.id_list[scan].append(view_ids)
                closest_idxs = []
                source_views = view_ids[1:]
                for vid in source_views:
                    closest_idxs.append(
                        get_nearest_pose_ids(
                            poses[vid, :, :],
                            ref_poses=poses[source_views],
                            num_select=5,
                            angular_dist_method="matrix",
                        )
                    )
                self.closest_idxs[scan].append(np.stack(closest_idxs, axis=0))

                c2w = np.eye(4).astype('float32')
                c2w[:3] = poses[idx]
                w2c = np.linalg.inv(c2w)
                self.c2ws[scan].append(c2w)
                self.w2cs[scan].append(w2c)

                co2w = np.eye(4).astype('float32')
                co2w[:3] = poses_other[idx]
                w2co = np.linalg.inv(co2w)
                self.co2ws[scan].append(co2w)
                self.w2cos[scan].append(w2co)

                intrinsic = refine_k_matrix.copy().astype('float32')
                self.intrinsics[scan].append(intrinsic)

    def __len__(self):
        return len(self.meta) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        if self.split == "train" and self.scene == "None":
            close_views = int(np.random.choice([3, 4, 5], 1))
            noisy_factor = 1.0
        else:
            noisy_factor = 1.0
            close_views = 5

        scan = self.meta[idx]["scan"]
        target_idx = self.meta[idx]["target_idx"]

        view_ids = self.id_list[scan][target_idx]
        target_view = view_ids[0]
        src_views = view_ids[1:]
        view_ids = [vid for vid in src_views] + [target_view]

        closest_idxs = self.closest_idxs[scan][target_idx][:, :close_views]

        imgs, imgs_other, depths, depths_other, depths_h, depths_aug = [], [], [], [], [], []
        intrinsics, w2cs, c2ws, w2cos, co2ws, near_fars = [], [], [], [], [], []
        affine_mats, affine_mats_inv, affine_other_mats, affine_other_mats_inv = [], [], [], []

        w, h = self.img_wh
        w, h = int(w * noisy_factor), int(h * noisy_factor)

        for vid in view_ids:
            img_filename = self.image_paths[scan][vid]
            img_other_filename = self.image_other_paths[scan][vid]
            depth_filename = self.depth_paths[scan][vid]
            depth_other_filename = self.depth_other_paths[scan][vid]
            img = Image.open(img_filename).convert("RGB")
            img_other = Image.open(img_other_filename).convert("RGB")
            with open(depth_filename, 'rb') as f:
                depth = pickle.load(f)
            with open(depth_other_filename, 'rb') as f:
                depth_other = pickle.load(f)    
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)
                img_other = img_other.resize((w, h), Image.BICUBIC)
                depth = cv2.resize(depth, dsize=(w, h), interpolation=cv2.INTER_AREA)
                depth_other = cv2.resize(depth_other, dsize=(w, h), interpolation=cv2.INTER_AREA)
            img = self.transform(img)
            img_other = self.transform(img_other)
            depth = torch.tensor(depth.copy())
            depth_other = torch.tensor(depth_other.copy())

            imgs.append(img)
            imgs_other.append(img_other)
            depths.append(depth)
            depths_other.append(depth_other)

            intrinsic = self.intrinsics[scan][vid].copy()
            intrinsic[:2] = intrinsic[:2] * noisy_factor
            intrinsics.append(intrinsic)

            w2c = self.w2cs[scan][vid]
            w2cs.append(w2c)
            c2ws.append(self.c2ws[scan][vid])

            w2co = self.w2cos[scan][vid]
            w2cos.append(w2co)
            co2ws.append(self.co2ws[scan][vid])

            aff = []
            aff_inv = []
            aff_other = []
            aff_other_inv = []
            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = intrinsic.copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2**l)
                proj_mat_l[:3, :4] = intrinsic_temp @ w2c[:3, :4]
                aff.append(proj_mat_l.copy())
                aff_inv.append(np.linalg.inv(proj_mat_l))

                proj_mat_l = np.eye(4)
                intrinsic_temp = intrinsic.copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2**l)
                proj_mat_l[:3, :4] = intrinsic_temp @ w2co[:3, :4]
                aff_other.append(proj_mat_l.copy())
                aff_other_inv.append(np.linalg.inv(proj_mat_l))

            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)
            aff_other = np.stack(aff_other , axis=-1)
            aff_other_inv = np.stack(aff_other_inv, axis=-1)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)
            affine_other_mats.append(aff_other)
            affine_other_mats_inv.append(aff_other_inv)

            valid_depth_mask = (0 < depth) * (depth < 10e2)
            depth_valid = depth[valid_depth_mask]
            valid_depth_other_mask = (0 < depth_other) * (depth_other < 10e2)
            depth_other_valid = depth[valid_depth_other_mask]
            near_fars.append([min(depth_valid.min().item(), depth_other_valid.min().item()), max(depth_valid.max().item(), depth_other_valid.max().item())])

        imgs = np.stack(imgs)
        imgs_other = np.stack(imgs_other)
        depths = np.stack(depths)
        depths_other = np.stack(depths_other)
        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        affine_other_mats = np.stack(affine_other_mats)
        affine_other_mats_inv = np.stack(affine_other_mats_inv)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)
        w2cos = np.stack(w2cos)
        co2ws = np.stack(co2ws)
        near_fars = np.stack(near_fars)
        baseline = np.linalg.norm(c2ws[0, :, -1] - co2ws[0, :, -1])

        sample = {}     

        ##########
        intrinsics = np.concatenate((intrinsics[:-1], intrinsics), axis=0)
        imgs = np.concatenate((imgs[:-1], imgs_other), axis=0)
        depths = np.concatenate((depths[:-1], depths_other), axis=0)
        w2cs = np.concatenate((w2cs[:-1], w2cos), axis=0)
        c2ws = np.concatenate((c2ws[:-1], co2ws), axis=0)
        near_fars = np.concatenate((near_fars[:-1], near_fars), axis=0)
        affine_mats = np.concatenate((affine_mats[:-1], affine_other_mats), axis=0)
        affine_mats_inv = np.concatenate((affine_mats_inv[:-1], affine_other_mats_inv), axis=0)
             
        closest_idxs = []
        for pose in c2ws[:-1]:
            closest_idxs.append(
                get_nearest_pose_ids(
                    pose, ref_poses=c2ws[:-1], num_select=5, angular_dist_method="matrix"
                )
            )
        closest_idxs = np.stack(closest_idxs, axis=0)[:, :close_views]

        depths_h = {}
        b, h, w = depths.shape
        for l in range(3):
            elem = np.zeros((b, int(h / (2**l)), int(w / (2**l))))
            for i in range(b):
                elem[i] = cv2.resize(
                    depths[i],
                    None,
                    fx=1.0 / (2**l),
                    fy=1.0 / (2**l),
                    interpolation=cv2.INTER_AREA,
                )
            depths_h[f"level_{l}"] = elem

        focal = intrinsics[:, 0, 0].astype('float32')
        baseline = np.array([self.baseline]).repeat(focal.size, axis=0).astype('float32')
        coef = (focal * baseline)[..., np.newaxis, np.newaxis]

        ##########
        sample["images"] = imgs
        sample["coef"] = coef
        sample["depths"] = depths
        sample["depths_h"] = depths_h
        sample["w2cs"] = w2cs
        sample["c2ws"] = c2ws
        sample["near_fars"] = near_fars
        sample["affine_mats"] = affine_mats
        sample["affine_mats_inv"] = affine_mats_inv
        sample["intrinsics"] = intrinsics
        sample["closest_idxs"] = closest_idxs
        sample["baseline"] = baseline
        sample["scan"] = scan

        return sample
