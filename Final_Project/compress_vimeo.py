import os
from PIL import Image


root = '/home/hentci/vimeo_triplet'
# 設定圖片來源資料夾和目標資料夾
with open(os.path.join(root, 'tri_testlist.txt'), 'r') as file:
    # 讀取每一行並去掉行尾的換行符號
    source_folders = [line.strip() for line in file]

quality = 10  # 設定JPEG壓縮品質（1-100），數值越低壓縮越高，畫質越低

for source_folder in source_folders:
    # 遍歷來源資料夾中的所有檔案
    for filename in os.listdir(os.path.join(root, 'sequences', source_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".JPG")):  # 只處理圖片檔案
            # 讀取圖片
            img_path = os.path.join(root, 'sequences', source_folder, filename)
            img = Image.open(img_path)

            # 設定JPEG壓縮格式並保存圖片到目標資料夾
            target_folder = os.path.join(root, 'comp', source_folder)
            os.makedirs(target_folder, exist_ok=True)
            save_path = os.path.join(target_folder, filename)
            img.save(save_path, 'JPEG', quality=quality)

            print(f"Compressed and saved {filename} to {save_path}")

print("JPEG compression completed.")