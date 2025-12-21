import builtins
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datetime import datetime
from PIL import Image
from pytorch_model_summary import summary
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.multiprocessing as mp

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True  # speed

class SupersamplingDataset(Dataset):
    def __init__(self, df, dataset_path, stage="train", augment=True):
        self.df = df.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.stage = stage  # 'train', 'val', 'test'
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def _apply_geometric_augment(self, lr_img, hr_img):
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img); hr_img = TF.hflip(hr_img)
        if random.random() < 0.5:
            lr_img = TF.vflip(lr_img); hr_img = TF.vflip(hr_img)

        return lr_img, hr_img


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]

        # Low-res image
        small_image_path = os.path.join(self.dataset_path, row["input_image"])
        small_image = Image.open(small_image_path).convert("RGB")

        # TRAIN or VAL: return LR + HR
        if self.stage in ("train", "val"):
            big_image_path = os.path.join(self.dataset_path, row["target_image"])
            big_image = Image.open(big_image_path).convert("RGB")

            # Apply joint augmentations (train only)
            if self.augment:
                small_image, big_image = self._apply_geometric_augment(small_image, big_image)

            # To tensor
            small_tensor = self.to_tensor(small_image)  # [3, 32, 32]
            big_tensor = self.to_tensor(big_image)      # [3, 128, 128]

            return img_id, small_tensor, big_tensor

        # TEST: only LR
        small_tensor = self.to_tensor(small_image)
        return img_id, small_tensor

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class UpsampleBlock(nn.Module):
    """2x upsample using PixelShuffle"""
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.ps   = nn.PixelShuffle(2)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class SRResNet4x(nn.Module):
    def __init__(self, num_blocks=12, channels=64, clamp_output=False):
        super().__init__()
        self.clamp_output = clamp_output

        self.conv_in = nn.Conv2d(3, channels, 3, 1, 1)
        self.act     = nn.LeakyReLU(0.2, inplace=True)

        self.blocks  = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(channels, channels, 3, 1, 1)

        self.up1 = UpsampleBlock(channels)  # 2x
        self.up2 = UpsampleBlock(channels)  # 4x total

        self.conv_out = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, lr):
        # bicubic skip
        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        x = self.act(self.conv_in(lr))
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x) + res

        x = self.up1(x)
        x = self.up2(x)
        x = self.conv_out(x)

        out = base + x  # residual learning

        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow.keys():
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{timestamp}] ", *args, **kwargs)

def get_timestamp():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def psnr(pred, target, eps=1e-10):
    """
    pred, target: (B, C, H, W) in [0,1]
    returns: average PSNR over batch (float)
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # (B,)
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def train(epoch_num):
    best_mse = np.inf
    ema = EMA(model)

    for epoch in range(epoch_num):
        start_time = time.time()
        model.train()

        for ids, lr, hr in train_dataloader:
            lr = lr.cuda(non_blocking=True)
            hr = hr.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss = criterion(pred, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            ema.update(model)
            scaler.update()

        scheduler.step()

        # ----- validation using EMA weights -----
        model.eval()
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.apply_to(model)

        val_mse = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        n = 0

        with torch.no_grad():
            for ids, lr, hr in validation_dataloader:
                lr = lr.cuda(non_blocking=True)
                hr = hr.cuda(non_blocking=True)

                pred = model(lr).clamp(0, 1)

                bs = lr.size(0)
                val_mse += criterion(pred, hr).item() * bs
                val_psnr += psnr(pred, hr) * bs
                val_ssim += ssim(pred, hr, data_range=1.0, size_average=True).item() * bs
                n += bs

        val_mse /= n
        val_psnr /= n
        val_ssim /= n

        # restore non-EMA weights for next epoch training
        model.load_state_dict(backup, strict=True)

        end_time = time.time()
        if val_mse < best_mse:
            best_mse = val_mse
        print(
            f"epoch {epoch + 1:3d}/{epoch_num:3d} | "
            f"MSE {val_mse:8.6f} | "
            f"PSNR {val_psnr:6.2f} dB | "
            f"SSIM {val_ssim:7.4f} | "
            f"time {end_time - start_time:5.1f} s | "
            f"lr {scheduler.get_last_lr()[0]:.1e}"
        )

    return best_mse


def create_submission_csv( model, test_loader, out_csv_path="submissions/submission.csv", device="cuda"):
    model.eval()
    model.to(device)

    H, W, C = 128, 128, 3
    num_pixels = H * W * C

    all_rows = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle (ids, lr) or (ids, lr, something_else)
            if len(batch) == 2:
                ids, lr_imgs = batch
            elif len(batch) == 3:
                ids, lr_imgs, _ = batch
            else:
                raise ValueError("Unexpected batch format from test_loader")

            lr_imgs = lr_imgs.to(device)           # [B, 3, 32, 32]

            # Forward pass: super-res prediction
            sr_imgs = model(lr_imgs)               # [B, 3, 128, 128]

            # Clamp to [0,1] before converting to [0,255]
            sr_imgs = sr_imgs.clamp(0.0, 1.0)

            # Move to CPU, (B, H, W, C)
            sr_np = sr_imgs.permute(0, 2, 3, 1).cpu().numpy()

            # Scale to [0,255], round and convert to uint8
            sr_np = (sr_np * 255.0).round().clip(0, 255).astype(np.uint8)

            B = sr_np.shape[0]
            for i in range(B):
                img_id = int(ids[i])
                # Flatten in row-major order -> (H * W * C,)
                flat_pixels = sr_np[i].reshape(-1)
                # Build row: [id, pixel_0, ..., pixel_49151]
                row = [img_id] + flat_pixels.tolist()
                all_rows.append(row)

    # Build DataFrame
    columns = ["id"] + [f"pixel_{i}" for i in range(num_pixels)]
    df = pd.DataFrame(all_rows, columns=columns)

    # Ensure sorted by id (just to be safe)
    df = df.sort_values("id").reset_index(drop=True)

    # Save
    df.to_csv(out_csv_path, index=False)
    print(f"Saved submission to {out_csv_path} with shape {df.shape}")

if __name__ == '__main__':
    mp.freeze_support()

    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test_input.csv")
    validation_df = pd.read_csv("dataset/validation.csv")
    batch_size = 128
    train_dataset = SupersamplingDataset(df=train_df, dataset_path="dataset/train/", stage="train", augment=True)

    validation_dataset = SupersamplingDataset( df=validation_df, dataset_path="dataset/validation/", stage="val", augment=False)

    test_dataset = SupersamplingDataset( df=test_df, dataset_path="dataset/test_input/", stage="test", augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True
                                  )
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True, persistent_workers=True
                                       )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=True
                                 )

    model = SRResNet4x(num_blocks=16, channels=96, clamp_output=False).cuda()
    nn.init.zeros_(model.conv_out.weight)
    if model.conv_out.bias is not None:
        nn.init.zeros_(model.conv_out.bias)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 300
    step = 60
    milestones = [milestone for milestone in range(step, num_epochs + 1, step)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    scaler = torch.amp.GradScaler('cuda')

    print(summary(model, torch.rand(size=(batch_size, 3, 32, 32)).cuda(), show_input=True))
    print("STARTED TRAINING")
    best_mse = train(epoch_num=num_epochs)
    print("FINISHED TRAINING")

    print("CREATING SUBMISSION CSV")
    create_submission_csv(
        model=model,
        test_loader=test_dataloader,
        out_csv_path=f"submissions/submission5-{best_mse:8.6f}.csv",
        device="cuda"
    )
    print("FINISHED")
    exit(0)