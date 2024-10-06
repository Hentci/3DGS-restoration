import os
from PIL import Image
from tqdm import tqdm

root = '/home/hentci/Final_Project/results/vim'
targets = ['vimeo_comp', 'vimeo_lq']
with open(os.path.join('/home/hentci/vimeo_triplet', 'tri_testlist.txt'), 'r') as file:
    # 讀取每一行並去掉行尾的換行符號
    folder_paths = [line.strip() for line in file]
    folder_paths = folder_paths[:100]

for target in targets:
    os.makedirs(os.path.join(root, 'simple', target), exist_ok=True)
    pbar = tqdm(folder_paths)
    pbar.set_description(target)
    for i, folder_path in enumerate(pbar):
        if i >= 100:
            break
        for imgname in os.listdir(os.path.join(root, target, folder_path)):
            img = Image.open(os.path.join(root, target, folder_path, imgname))
            img.save(os.path.join(root, 'simple', target, folder_path.replace('/', '')+imgname), quality=100)