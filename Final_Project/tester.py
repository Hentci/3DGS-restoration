# implement your training script here
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import Vim

from cProfile import label
import torch.utils
import torch.utils.data
import numpy as np
import os
import torch.nn.functional as F
from utils import utils_blindsr as blindsr
from PIL import Image
from torchvision.utils import save_image
import torchmetrics
import torchvision.transforms as transforms

class VimeoDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        # 打開並讀取 txt 檔案
        with open(os.path.join(root, 'tri_testlist.txt'), 'r') as file:
            # 讀取每一行並去掉行尾的換行符號
            self.folder_paths = [line.strip() for line in file]
            self.folder_paths = self.folder_paths[:100]

        self.root = root
        
    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        lq_path = os.path.join(self.root, 'comp', self.folder_paths[idx])
        hq_path = os.path.join(self.root, 'sequences', self.folder_paths[idx])
        filenames = os.listdir(lq_path)
        
        # 取得前三個檔案的完整路徑
        lq_image_paths = [os.path.join(lq_path, filename) for filename in sorted(filenames)[:3]]
        hq_image_paths = [os.path.join(hq_path, filename) for filename in sorted(filenames)[:3]]
        
        # 讀取圖片並轉換為 numpy array，並確保形狀是 (H, W, C)
        img_lqs = [np.array(Image.open(image_path).convert('RGB').resize((256, 256))) / 255.0 for image_path in lq_image_paths]
        img_hqs = [np.array(Image.open(image_path).convert('RGB')) / 255.0 for image_path in hq_image_paths]
        
        # 將結果轉換為所需的形狀 (3, C, H, W)
        img_lq = np.stack([np.transpose(img_lq, (2, 0, 1)) for img_lq in img_lqs], axis=0)
        img_hq = np.stack([np.transpose(img_hq, (2, 0, 1)) for img_hq in img_hqs], axis=0)
        
        # 將 numpy array 轉換為 tensor
        img_lq = torch.from_numpy(img_lq).float()
        img_hq = torch.from_numpy(img_hq).float()
        
        return img_lq, img_hq, self.folder_paths[idx], filenames


class BicycleDataset(torch.utils.data.Dataset):
    def __init__(self, root='/home/hentci/mipnerf360_dataset/', img_size=512):
        self.input_path = os.path.join(root, 'lq')
        self.gt_path = os.path.join(root, 'bicycle/images_4')
        self.filenames = sorted(os.listdir(self.input_path))
        self.root = root
        self.img_size = img_size
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
        ])
        gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        inputs = torch.stack([input_transform(Image.open(os.path.join(self.input_path, self.filenames[idx])))])
        gt = torch.stack([gt_transform(Image.open(os.path.join(self.gt_path, self.filenames[idx])))])
        
        return inputs, gt, 'bicycle_bsr_4', [self.filenames[idx]]

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--root', type=str, default='../vimeo_triplet')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--num_worker', type=int, default=8, help='Number Workers')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--ckpt_path', type=str, default='./saved_models/epoch_20.pth')
    parser.add_argument('--result_dir', type=str, default='./results/vim/vimeo_comp')
    parser.add_argument('--img_size', type=int, default=256)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    model = Vim(
        dim=256,  # Dimension of the transformer model
        # heads=8,  # Number of attention heads
        dt_rank=32,  # Rank of the dynamic routing matrix
        dim_inner=256,  # Inner dimension of the transformer model
        d_state=256,  # Dimension of the state vector
        image_size=args.img_size,  # Size of the input image
        patch_size=16,  # Size of each image patch
        channels=3,  # Number of input channels
        dropout=0.1,  # Dropout rate
        depth=12,  # Depth of the transformer model
    ).to(device=args.device)
    
    test_dataset = VimeoDataset(args.root)
    # test_dataset = BicycleDataset(root='/home/hentci/mipnerf360_dataset/', img_size=args.img_size)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=args.num_worker)
    
    
    print(len(test_loader))
    
    last_epoch = 0
    if args.ckpt_path != '':
        saved_ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(saved_ckpt['model'])
    
    model.eval()
    
    pbar = tqdm(test_loader)
    pbar.set_description(f'Test')
    psnrs = []
    ssims = []
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            inputs, gt, folder, filenames = data
            inputs = inputs.to(device=args.device)
            gt = gt.to(device=args.device)
            
            B, S, C, H, W = gt.shape
            
            y = model(inputs)
            
            gt = gt.reshape(B*S, C, H, W)
            y = y.reshape(B*S, C, args.img_size, args.img_size)
            y = transforms.Resize((H, W))(y)
            
            filenames = [item[0] for item in filenames]
            for i in range(B*S):
                os.makedirs(os.path.join(args.result_dir, folder[i//S]), exist_ok=True)
                # save_image(y[i], os.path.join(args.result_dir, folder[i//S], filenames[i]))
                img = transforms.ToPILImage()(y[i].clamp(0, 1))
                img.save(os.path.join(args.result_dir, folder[i//S], filenames[i]), quality=100)
            
            psnr = torchmetrics.functional.peak_signal_noise_ratio(y, gt, data_range=1.0)
            ssim = torchmetrics.functional.structural_similarity_index_measure(y, gt, data_range=1.0)
            
            psnrs.append(psnr.cpu().numpy())
            ssims.append(ssim.cpu().numpy())
            pbar.set_postfix_str(f'PSNR: {psnr:.3f}, SSIM: {ssim:.3f}')
        
        print(f'PSNR: {np.mean(psnrs):.3f}, SSIM: {np.mean(ssims):.3f}')
        
            
        
        

