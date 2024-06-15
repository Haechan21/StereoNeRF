import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from tqdm import tqdm
import math
import argparse

import lpips
lpips_fn = lpips.LPIPS(net="vgg")

mse2psnr = lambda x: -10.0 * np.log(x) / np.log(10)

def img2mse(img_pr, img_gt, mask=False):
    h, w, c = img_pr.shape
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    if mask:
        mask = np.sum(img_pr, axis=1)
        mask = (mask != 0).flatten()
        img_gt[~mask] = 0
        img_pr[~mask] = 0
    
    return np.mean((img_gt[mask] - img_pr[mask]) ** 2), img_pr.reshape(h, w, c), img_gt.reshape(h, w, c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    args = parser.parse_args()

    base_dir = args.pred
    dir_list = os.listdir(base_dir)
    
    psnr_all = 0
    ssim_all = 0
    lpips_all = 0
    cnt = 0

    for scene in tqdm(dir_list):
        if not scene.startswith('00000000'):
            continue

        infer_res = cv2.imread(base_dir + scene) / 255.
        h, w_4, c = infer_res.shape
        w = int(w_4 / 4) 
        
        gt, fine_res = infer_res[:, :w, :], infer_res[:, w:2*w, :]
        
        mse, fine_res, gt = img2mse(fine_res, gt, mask=True)
        cur_psnr = mse2psnr(mse)
        cur_ssim = ssim(gt, fine_res, data_range=1, multichannel=True)
        
        gt_lpips, fine_res = torch.tensor(gt.transpose(2, 0, 1), dtype=torch.float32), torch.tensor(fine_res.transpose(2, 0, 1), dtype=torch.float32)
        gt_lpips, fine_res = gt_lpips * 2 - 1, fine_res * 2 - 1
        cur_lpips = lpips_fn(gt_lpips.unsqueeze(0), fine_res.unsqueeze(0)).item()
        
        psnr_all += cur_psnr
        ssim_all += cur_ssim
        lpips_all += cur_lpips
        cnt += 1
     
    psnr_mean = math.ceil(psnr_all / cnt * 100) / 100
    ssim_mean = math.ceil(ssim_all / cnt * 10000) /10000
    lpips_mean = math.floor(lpips_all / cnt * 10000) / 10000
    
    print("Cnt :", cnt)
    print("PSNR :", psnr_mean)
    print("SSIM :", ssim_mean)
    print("LPIPS :", lpips_mean)