from PIL import Image
import numpy as np
import os
from utils import utils_blindsr as blindsr
from tqdm import tqdm

if __name__ == '__main__':
    root = '../vimeo_triplet'
    # 打開並讀取 txt 檔案
    with open(os.path.join(root, 'tri_testlist.txt'), 'r') as file:
        # 讀取每一行並去掉行尾的換行符號
        folder_paths = [line.strip() for line in file]
        
    pbar = tqdm(folder_paths)

    for _, folder in enumerate(pbar):
        folder_path = os.path.join(root, 'sequences', folder)
        lq_path = os.path.join(root, 'lq', folder)
        hq_path = os.path.join(root, 'hq', folder)
        filenames = os.listdir(folder_path)
        
        # 取得前三個檔案的完整路徑
        image_paths = [os.path.join(folder_path, filename) for filename in sorted(filenames)[:3]]
        
        # 讀取圖片並轉換為 numpy array，並確保形狀是 (H, W, C)
        imgs = [np.array(Image.open(image_path).convert('RGB').resize((256, 256))) / 255.0 for image_path in image_paths]
        
        # 使用 degradation 函式分別處理每一張圖片
        img_lqs, img_hqs = zip(*[blindsr.degradation_bsrgan_plus(img, sf=1, shuffle_prob=0.1, use_sharp=True, lq_patchsize=256) for img in imgs])
        
        # 將結果轉換為所需的形狀 (3, C, H, W)
        img_lqs = [(img_lq*255).astype(np.uint8) for img_lq in img_lqs]
        img_hqs = [(img_hq*255).astype(np.uint8) for img_hq in img_hqs]
        
        # save degrad images
        os.makedirs(lq_path, exist_ok=True)
        os.makedirs(hq_path, exist_ok=True)
        for i, filename in enumerate(filenames):
            img = Image.fromarray(img_lqs[i])
            img.save(os.path.join(lq_path, filename))
            # print(f'save {os.path.join(lq_path, filename)}')
            
            img = Image.fromarray(img_hqs[i])
            img.save(os.path.join(hq_path, filename))
            # print(f'save {os.path.join(hq_path, filename)}')