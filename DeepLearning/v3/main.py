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

import wandb
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

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc2(self.act(self.fc1(w)))
        return x * self.gate(w)

class RCAB(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1, reduction: int = 16):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act   = nn.ReLU(inplace=True)  # ReLU is typical for PSNR/MSE SR
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.ca    = SEBlock(channels, reduction=reduction)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        res = self.ca(res)
        return x + self.residual_scale * res

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, residual_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.residual_scale = residual_scale

    def forward(self, x):
        return x + self.residual_scale * self.conv2(self.act(self.conv1(x)))

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

class SRAttentionResNet4x(nn.Module):
    def __init__(self, num_blocks=24, channels=320, residual_scale=0.1, clamp_output=False, reduction=16):
        super().__init__()
        self.clamp_output = clamp_output
        self.residual_scale = residual_scale

        self.conv_in = nn.Conv2d(3, channels, 3, 1, 1)
        self.act     = nn.ReLU(inplace=True)  # you can keep this, or switch to ReLU

        self.blocks = nn.Sequential(*[
            RCAB(channels, residual_scale=residual_scale, reduction=reduction)
            for _ in range(num_blocks)
        ])
        self.conv_mid = nn.Conv2d(channels, channels, 3, 1, 1)

        self.up1 = UpsampleBlock(channels)
        self.up2 = UpsampleBlock(channels)
        self.conv_out = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, lr):
        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        x = self.act(self.conv_in(lr))
        trunk_in = x

        x = self.blocks(x)
        x = self.conv_mid(x)

        # trunk residual scaling (important when wide/deep)
        x = trunk_in + self.residual_scale * x

        x = self.up1(x)
        x = self.up2(x)
        x = self.conv_out(x)

        out = base + x
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

def train(epoch_num, run=None):
    patience = 10
    min_delta = 1e-7
    epochs_no_improve = 0
    best_mse = np.inf

    ema = EMA(model)

    global_step = 0

    for epoch in range(epoch_num):
        start_time = time.time()
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        # ---- train ----
        for ids, lr, hr in train_dataloader:
            lr = lr.cuda(non_blocking=True)
            hr = hr.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred = model(lr)
                loss = criterion(pred, hr)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            ema.update(model)
            scaler.update()

            bs = lr.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            # optional: log a few step-level stats (keeps charts smooth)
            if run is not None and (global_step % 50 == 0):
                wandb.log({
                    "train/step_mse": loss.item(),
                    "train/grad_norm": float(grad_norm),
                }, step=global_step)

            global_step += 1

        scheduler.step()

        train_mse = train_loss_sum / max(train_n, 1)

        # ---- validation on EMA weights ----
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

        # restore non-EMA weights
        model.load_state_dict(backup, strict=True)

        end_time = time.time()
        lr_now = scheduler.get_last_lr()[0]

        improved = val_mse < best_mse - min_delta
        if improved:
            best_mse = val_mse
            epochs_no_improve = 0
            torch.save(ema.shadow, "best_ema.pth")
        else:
            epochs_no_improve += 1

        # ---- wandb epoch logging ----
        if run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "lr": lr_now,
                "time/epoch_sec": end_time - start_time,
                "train/mse": train_mse,
                "val/mse": val_mse,
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
            }, step=global_step)

        print(
            f"epoch {epoch + 1:3d}/{epoch_num:3d} | "
            f"train_mse {train_mse:8.6f} | "
            f"val_mse {val_mse:8.6f} | "
            f"PSNR {val_psnr:6.2f} dB | "
            f"SSIM {val_ssim:7.4f} | "
            f"time {end_time - start_time:5.1f} s | "
            f"lr {lr_now:.1e}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best MSE: {best_mse:.6f}")
            break

    return best_mse

def _apply_t(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply one of 8 transforms to a BCHW tensor.
    k in [0..7]:
      - rot = k % 4  (0, 90, 180, 270)
      - if k >= 4: horizontal flip
    """
    # Horizontal flip (W axis)
    if k >= 4:
        x = torch.flip(x, dims=[3])

    # Rotate on (H, W)
    rot = k % 4
    if rot != 0:
        x = torch.rot90(x, k=rot, dims=[2, 3])

    return x

def _invert_t(y: torch.Tensor, k: int) -> torch.Tensor:
    """Invert the transform applied by _apply_t."""
    rot = k % 4
    if rot != 0:
        y = torch.rot90(y, k=4 - rot, dims=[2, 3])

    if k >= 4:
        y = torch.flip(y, dims=[3])

    return y

@torch.no_grad()
def self_ensemble_sr(model: torch.nn.Module, lr_imgs: torch.Tensor) -> torch.Tensor:
    """
    x8 self-ensemble:
      average_{k=0..7} inv_t( model( apply_t(lr, k) ), k )
    """
    preds = []
    for k in range(8):
        inp = _apply_t(lr_imgs, k)
        out = model(inp)
        out = _invert_t(out, k)
        preds.append(out)
    return torch.stack(preds, dim=0).mean(dim=0)

def create_submission_csv(model, test_loader, out_csv_path="submissions/submission.csv", device="cuda"):
    """
    Creates Kaggle submission:
      id,pixel_0,...,pixel_49151
    where pixels are uint8 0..255 in row-major order with RGB consecutive.

    Uses best EMA weights (best_ema.pth) and x8 self-ensemble by default.
    """
    model.load_state_dict(torch.load("best_ema.pth", map_location=device), strict=True)
    model.to(device)
    model.eval()

    H, W, C = 128, 128, 3
    num_pixels = H * W * C  # 49152

    all_rows = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                ids, lr_imgs = batch
            elif len(batch) == 3:
                ids, lr_imgs, _ = batch
            else:
                raise ValueError("Unexpected batch format from test_loader")

            lr_imgs = lr_imgs.to(device, non_blocking=True)  # (B,3,32,32)

            # ---- x8 self-ensemble SR ----
            sr_imgs = self_ensemble_sr(model, lr_imgs)       # (B,3,128,128)

            # Clamp and convert to uint8 pixels
            sr_imgs = sr_imgs.clamp(0.0, 1.0)
            sr_uint8 = (sr_imgs * 255.0).round().clamp(0, 255).to(torch.uint8)  # (B,3,128,128)

            # Flatten in correct order: (H,W,C) row-major, RGB consecutive
            flat = (
                sr_uint8.permute(0, 2, 3, 1)                 # (B,128,128,3)
                       .reshape(sr_uint8.size(0), -1)        # (B,49152)
                       .cpu()
                       .numpy()
            )

            for i in range(flat.shape[0]):
                img_id = int(ids[i])
                all_rows.append([img_id] + flat[i].tolist())

    columns = ["id"] + [f"pixel_{i}" for i in range(num_pixels)]
    df = pd.DataFrame(all_rows, columns=columns).sort_values("id").reset_index(drop=True)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved submission to {out_csv_path} with shape {df.shape}")

def main_run(config):
    global model, criterion, optimizer, scheduler, scaler
    global train_dataloader, validation_dataloader, test_dataloader

    # ---- wandb init ----
    run = wandb.init(
        project="DeepLearning-v3",
        name=config["run_name"],
        config=config
    )

    # Optional: log gradients/params (can slow on big nets)
    # wandb.watch(model, log="gradients", log_freq=200)

    # ---- build model ----
    if config["model_type"] == "plain":
        model = SRResNet4x(
            num_blocks=config["num_blocks"],
            channels=config["channels"],
            clamp_output=False
        ).cuda()
    elif config["model_type"] == "attn":
        model = SRAttentionResNet4x(
            num_blocks=config["num_blocks"],
            channels=config["channels"],
            residual_scale=config["residual_scale"],
            reduction=config["reduction"],
            clamp_output=False
        ).cuda()
    else:
        raise ValueError("Unknown model_type")

    print(summary(model, torch.rand(size=(batch_size, 3, 32, 32)).cuda(), show_input=True))

    nn.init.zeros_(model.conv_out.weight)
    if model.conv_out.bias is not None:
        nn.init.zeros_(model.conv_out.bias)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
        eta_min=config["eta_min"]
    )

    scaler = torch.amp.GradScaler('cuda')

    print("STARTED TRAINING")
    best_mse = train(epoch_num=config["num_epochs"], run=run)
    print("FINISHED TRAINING")

    # Save best as artifact (optional)
    artifact = wandb.Artifact(f"best_ema_{config['run_name']}", type="model")
    artifact.add_file("best_ema.pth")
    run.log_artifact(artifact)

    # Make submission (logs best_mse into filename)
    print("CREATING SUBMISSION CSV")
    out_path = f"submissions/submission-{config['run_name']}-{best_mse:8.6f}.csv"
    create_submission_csv(model=model, test_loader=test_dataloader, out_csv_path=out_path, device="cuda")

    run.finish()
    return best_mse


if __name__ == '__main__':
    mp.freeze_support()

    with open("api_key.txt", "r") as f:
        api_key = f.readline().strip()
        wandb.login(api_key)

    batch_size = 16
    num_workers = 8
    train_df = pd.read_csv("dataset/train.csv")
    test_df = pd.read_csv("dataset/test_input.csv")
    validation_df = pd.read_csv("dataset/validation.csv")
    train_dataset = SupersamplingDataset(df=train_df, dataset_path="dataset/train/", stage="train", augment=True)
    validation_dataset = SupersamplingDataset( df=validation_df, dataset_path="dataset/validation/", stage="val", augment=False)
    test_dataset = SupersamplingDataset( df=test_df, dataset_path="dataset/test_input/", stage="test", augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)


    sweep = [
        {"run_name":  "plain_b8_c896", "model_type": "plain", "num_blocks": 8,  "channels": 896},
        {"run_name": "plain_b16_c896", "model_type": "plain", "num_blocks": 16, "channels": 896},
        {"run_name": "plain_b24_c896", "model_type": "plain", "num_blocks": 24, "channels": 896},

        # {"run_name":  "attn_b8_c896", "model_type": "attn", "num_blocks": 8,  "channels": 612, "residual_scale": 0.1, "reduction": 1},
        # {"run_name": "attn_b16_c612", "model_type": "attn", "num_blocks": 16, "channels": 612, "residual_scale": 0.1, "reduction": 1},
        # {"run_name": "attn_b24_c612", "model_type": "attn", "num_blocks": 24, "channels": 612, "residual_scale": 0.1, "reduction": 1},
    ]

    base = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": 75,
        "lr": 1e-4,
        "eta_min": 1e-7,
        "weight_decay": 0.0,
    }

    for cfg in sweep:
        config = {**base, **cfg}
        best_mse = main_run(config)
        print("DONE. Best EMA val MSE:", best_mse)
