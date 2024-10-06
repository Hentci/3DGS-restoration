# implement your training script here
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
from numpy import mean
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import VimeoDataset
from model import Vim

import tensorboard as tb
from torch.utils.tensorboard.writer import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--root', type=str, default='../vimeo_triplet')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_worker', type=int, default=8, help='Number Workers')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--save_dir', type=str, default='./saved_models/')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./saved_models/epoch_10.pth')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    model = Vim(
        dim=256,  # Dimension of the transformer model
        # heads=8,  # Number of attention heads
        dt_rank=32,  # Rank of the dynamic routing matrix
        dim_inner=256,  # Inner dimension of the transformer model
        d_state=256,  # Dimension of the state vector
        image_size=256,  # Size of the input image
        patch_size=16,  # Size of each image patch
        channels=3,  # Number of input channels
        dropout=0.1,  # Dropout rate
        depth=12,  # Depth of the transformer model
    ).to(device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
    
    train_dataset = VimeoDataset(args.root)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.num_worker)
    
    
    print(len(train_loader))
    
    last_epoch = 0
    if args.ckpt_path != '':
        saved_ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(saved_ckpt['model'])
        optimizer.load_state_dict(saved_ckpt['optim'])
        scheduler.load_state_dict(saved_ckpt['sched'])
        last_epoch = saved_ckpt['epoch']
    
    writer = SummaryWriter()
    best_loss = float('inf')
    for epoch in range(last_epoch+1, args.epochs+1):
        model.train()
        
        pbar = tqdm(train_loader)
        pbar.set_description(f'Epoch {epoch}')
        losses = []
        for i, data in enumerate(pbar):
            optimizer.zero_grad()
            
            inputs, gt = data
            inputs = inputs.to(device=args.device)
            gt = gt.to(device=args.device)
            
            y = model(inputs)
            
            loss = torch.nn.functional.mse_loss(y, gt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix_str(f'Loss: {loss.item()}', refresh=True)
        
        scheduler.step(mean(losses))
        writer.add_scalar(f'Loss', mean(losses), epoch)
        
        os.makedirs(args.save_dir, exist_ok=True)
        if mean(losses) < best_loss:
            best_loss = mean(losses)
            torch.save({'model':model.state_dict(),
                        'optim':optimizer.state_dict(),
                        'sched':scheduler.state_dict(),
                        'epoch':epoch}, os.path.join(args.save_dir, f'best_loss.pth'))
        
        if epoch % args.save_freq == 0:
            torch.save({'model':model.state_dict(),
                        'optim':optimizer.state_dict(),
                        'sched':scheduler.state_dict(),
                        'epoch':epoch}, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))
    
    writer.close()
        

