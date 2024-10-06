import os
from PIL import Image

# 設定圖片來源資料夾和目標資料夾
source_folder = '/home/hentci/mipnerf360_dataset/bicycle/images_4'
target_folder = '/home/hentci/mipnerf360_dataset/comp3_bicycle/images_4/'
quality = 3  # 設定JPEG壓縮品質（1-100），數值越低壓縮越高，畫質越低

# 檢查目標資料夾是否存在，不存在則創建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍歷來源資料夾中的所有檔案
for filename in os.listdir(source_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".JPG")):  # 只處理圖片檔案
        # 讀取圖片
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)

        # 設定JPEG壓縮格式並保存圖片到目標資料夾
        save_path = os.path.join(target_folder, filename)
        img.save(save_path, 'JPEG', quality=quality)

        print(f"Compressed and saved {filename} to {save_path}")

print("JPEG compression completed.")