# implement your training script here
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import torchvision.transforms as transforms

# 初始化 LPIPS 模型
loss_fn = lpips.LPIPS(net='vgg').cuda()  # 使用 'alex' backbone

def calculate_metrics(img1, img2):
    # 計算 PSNR 和 SSIM
    img1_np = np.array(img1) / 255.0
    img2_np = np.array(img2) / 255.0
    
    psnr_value = psnr(img1_np, img2_np, data_range=1.)
    ssim_value = ssim(img1_np, img2_np, channel_axis=2, data_range=1.)

    # 計算 LPIPS
    img1_tensor = transforms.ToTensor()(img1).unsqueeze(0).cuda()
    img2_tensor = transforms.ToTensor()(img2).unsqueeze(0).cuda()
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()

    return psnr_value, ssim_value, lpips_value

def compare_folders(folder1, folder2):
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for filename in os.listdir(folder1):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        if os.path.isfile(img1_path) and os.path.isfile(img2_path):
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            psnr_value, ssim_value, lpips_value = calculate_metrics(img1, img2)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')
    print(f'Average LPIPS: {avg_lpips}')

# 指定資料夾路徑
folder1 = '/home/hentci/vimeo_triplet/simple/sequences'
folder2 = '/home/hentci/Final_Project/results/vim/simple/vimeo_comp'

# 計算並顯示結果
compare_folders(folder1, folder2)