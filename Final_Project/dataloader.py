import torch
import torch.utils
import torch.utils.data
import numpy as np
import os
import torch.nn.functional as F
from utils import utils_blindsr as blindsr
from PIL import Image


class VimeoDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        # 打開並讀取 txt 檔案
        with open(os.path.join(root, 'tri_trainlist.txt'), 'r') as file:
            # 讀取每一行並去掉行尾的換行符號
            self.folder_paths = [line.strip() for line in file]

        self.root = root
        
    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.root, 'sequences', self.folder_paths[idx])
        filenames = os.listdir(folder_path)
        
        # 取得前三個檔案的完整路徑
        image_paths = [os.path.join(folder_path, filename) for filename in sorted(filenames)[:3]]
        
        # 讀取圖片並轉換為 numpy array，並確保形狀是 (H, W, C)
        imgs = [np.array(Image.open(image_path).convert('RGB').resize((256, 256))) / 255.0 for image_path in image_paths]
        
        
        # 使用 degradation 函式分別處理每一張圖片
        img_lqs, img_hqs = zip(*[blindsr.degradation_bsrgan_plus(img, sf=1, shuffle_prob=0.1, use_sharp=True, lq_patchsize=256) for img in imgs])
        
        
        # 將結果轉換為所需的形狀 (3, C, H, W)
        img_lq = np.stack([np.transpose(img_lq, (2, 0, 1)) for img_lq in img_lqs], axis=0)
        img_hq = np.stack([np.transpose(img_hq, (2, 0, 1)) for img_hq in img_hqs], axis=0)
        
        # 將 numpy array 轉換為 tensor
        img_lq = torch.from_numpy(img_lq).float()
        img_hq = torch.from_numpy(img_hq).float()
        
        return img_lq, img_hq
        