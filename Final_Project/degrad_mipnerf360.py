from PIL import Image
import numpy as np
import os
from utils import utils_blindsr as blindsr
from tqdm import tqdm

if __name__ == '__main__':
    root = '/home/hentci/mipnerf360_dataset'
    input_path = os.path.join(root, 'bicycle/images_4')
    filenames = sorted(os.listdir(input_path))
        
    pbar = tqdm(filenames)

    for _, filename in enumerate(pbar):
        image_path = os.path.join(input_path, filename)
        lq_path = os.path.join(root, 'lq')
        
        # 讀取圖片並轉換為 numpy array，並確保形狀是 (H, W, C)
        img = np.array(Image.open(image_path).convert('RGB').resize((512, 512))) / 255.0
        
        # 使用 degradation 函式分別處理每一張圖片
        img_lq, img_hq = blindsr.degradation_bsrgan_plus(img, sf=1, shuffle_prob=0.1, use_sharp=True, lq_patchsize=512)
        
        # 將結果轉換為所需的形狀 (3, C, H, W)
        img_lq = (img_lq*255).astype(np.uint8)
        img_hq = (img_hq*255).astype(np.uint8)
        
        # save degrad images
        os.makedirs(lq_path, exist_ok=True)
        img = Image.fromarray(img_lq).resize((1237, 822))
        img.save(os.path.join(lq_path, filename))
        # print(f'save {os.path.join(lq_path, filename)}')
            