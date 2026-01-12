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


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    )

class BicubicRefiner(nn.Module):
    def __init__(self, num_channels, num_blocks):
        super().__init__()
        layers = []

        # Feature extraction on HR images
        layers.append(self.conv_layer(3, num_channels, kernel_size=3, stride=1, padding=1))

        for _ in range(num_blocks):
            layers.append(self.conv_layer(num_channels, num_channels, kernel_size=3, stride=1, padding=1))

        # Reconstruction to RGB image layer
        layers.append(nn.Conv2d(num_channels, 3, 3, 1, 1))

        self.head = nn.Sequential(*layers[:-1])  # split head and conv_out to solve an error of pytorch_model_summary
        self.conv_out = layers[-1]

    def _conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, lr):

        base = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        # predict residual correction on HR img
        res = self.conv_out(self.head(base))
        out = base + res

        return out

class SrNet(nn.Module):
    def __init__(
        self,
        blocks: int = 12,
        channels: int = 64,
    ):
        super().__init__()
        self.blocks = blocks
        self.channels = channels
        self.residual_scale = 0.1

        # 1. Feature extraction (b, 3, 32, 32) -> (b, channels, 32, 32)
        self.feature_extraction = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 2. Residual Blocks -> keep same dimension ( b, channels, 32, 32 )
        self.residual_blocks = nn.ModuleList([self._residual_block() for _ in range(blocks)])

        # 3. Upsample Blocks -> (b, channels, 32, 32) -> (b, channels, 64, 64) -> (b, channels, 128, 128)
        self.upsample1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        # 4. SR Img reconstruction
        self.sr_img_reconstruction = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)

    def _residual_block(self):
        return nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, lr_img):
        # 1. Feature Extraction
        x = self.feature_extraction(lr_img)
        x = self.activation(x)

        # 2. Residual Blocks
        for block in self.residual_blocks:
            x = x + self.residual_scale * block(x)

        # 3. Upsample blocks
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upsample1(x)
        x = self.activation(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upsample2(x)
        x = self.activation(x)

        # 4. Sr Img Reconstruction
        sr_img = self.sr_img_reconstruction(x)
        return sr_img

def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{timestamp}] ", *args, **kwargs)


def get_timestamp():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # (B,)
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()


def train(epoch_num, run=None):
    patience = 10
    min_delta = 1e-7
    epochs_no_improve = 0
    best_mse = np.inf

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
            scaler.update()

            bs = lr.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            if run is not None and (global_step % 50 == 0):
                wandb.log({
                    "train/step_mse": loss.item(),
                    "train/grad_norm": float(grad_norm),
                }, step=global_step)

            global_step += 1

        scheduler.step()
        train_mse = train_loss_sum / max(train_n, 1)

        # ---- validation ----
        model.eval()
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

        end_time = time.time()
        lr_now = scheduler.get_last_lr()[0]

        # ---- checkpoint / early stopping ----
        if val_mse < best_mse - min_delta:
            best_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")  # <--- save state_dict
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


def create_submission_csv(model, test_loader, out_csv_path="submissions/submission.csv", device="cuda", weights_path="best_model.pth"):
    # Load weights
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    H, W, C = 128, 128, 3
    num_pixels = H * W * C  # 49152

    all_rows = []

    with torch.no_grad():
        for batch in test_loader:
            # batch can be (ids, lr) or (ids, lr, hr/anything)
            ids, lr_imgs = batch[0], batch[1]

            lr_imgs = lr_imgs.to(device, non_blocking=True)   # (B,3,32,32)

            # Single forward pass
            sr_imgs = model(lr_imgs)                          # (B,3,128,128)

            # Clamp -> uint8
            sr_imgs = sr_imgs.clamp(0.0, 1.0)
            sr_uint8 = (sr_imgs * 255.0).round().clamp(0, 255).to(torch.uint8)  # (B,3,128,128)

            # Flatten in row-major order with RGB consecutive: (H,W,C)
            flat = (
                sr_uint8.permute(0, 2, 3, 1)   # (B,128,128,3)
                        .reshape(sr_uint8.size(0), -1)  # (B,49152)
                        .cpu()
                        .numpy()
            )

            for i in range(flat.shape[0]):
                all_rows.append([int(ids[i])] + flat[i].tolist())

    columns = ["id"] + [f"pixel_{i}" for i in range(num_pixels)]
    df = pd.DataFrame(all_rows, columns=columns)
    df = df.sort_values("id").reset_index(drop=True)
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

    # ---- build model ----
    if config["model_type"] == "srnet":
        model = SrNet(
            blocks=config["num_blocks"],
            channels=config["channels"]
        ).cuda()
    elif config["model_type"] == "refiner":
        ValueError("add model type")
        model = BicubicRefiner(
            num_channels=config["channels"],
            num_blocks=config["num_blocks"]
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

    # Save best as artifact
    artifact = wandb.Artifact(f"best_model_{config['run_name']}", type="model")
    artifact.add_file("best_model.pth")
    run.log_artifact(artifact)

    # Make submission (logs best_mse into filename)
    # print("CREATING SUBMISSION CSV")
    # out_path = f"submissions/submission-{config['run_name']}-{best_mse:8.6f}.csv"
    # create_submission_csv(model=model, test_loader=test_dataloader, out_csv_path=out_path, device="cuda")

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
        # --- 75-epoch cosine ---
        # {"run_name": "srnet_b24_c712_lr3e-5_e75", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 3e-5, "num_epochs": 75},
        # {"run_name": "srnet_b24_c712_lr5e-5_e75", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 5e-5, "num_epochs": 75},
        #  {"run_name": "srnet_b24_c712_lr1e-4_e75", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 1e-4, "num_epochs": 75},
        #
        # # --- 100-epoch cosine ---
        # {"run_name": "srnet_b24_c712_lr3e-5_e100", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 3e-5, "num_epochs": 100},
        # {"run_name": "srnet_b24_c712_lr5e-5_e100", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 5e-5, "num_epochs": 100},
        # {"run_name": "srnet_b24_c712_lr7e-5_e100", "model_type": "srnet", "num_blocks": 24, "channels": 712, "lr": 7.5e-5, "num_epochs": 100},

        # num_blocks = 8
        # {"run_name": "refiner_b8_c64", "model_type": "refiner", "num_blocks": 8, "channels":  64},
        # {"run_name": "refiner_b8_c128", "model_type": "refiner", "num_blocks": 8, "channels": 128},
        # {"run_name": "refiner_b8_c192", "model_type": "refiner", "num_blocks": 8, "channels": 192},
        # {"run_name": "refiner_b8_c256", "model_type": "refiner", "num_blocks": 8, "channels": 256},
        # {"run_name": "refiner_b8_c320", "model_type": "refiner", "num_blocks": 8, "channels": 320},
        # {"run_name": "refiner_b8_c384", "model_type": "refiner", "num_blocks": 8, "channels": 384},
        # {"run_name": "refiner_b8_c448", "model_type": "refiner", "num_blocks": 8, "channels": 448},
        # {"run_name": "refiner_b8_c512", "model_type": "refiner", "num_blocks": 8, "channels": 512},
        # {"run_name": "refiner_b8_c576", "model_type": "refiner", "num_blocks": 8, "channels": 576},
        # {"run_name": "refiner_b8_c640", "model_type": "refiner", "num_blocks": 8, "channels": 640},
        # {"run_name": "refiner_b8_c712", "model_type": "refiner", "num_blocks": 8, "channels": 712},

        {"run_name": "refiner_b16_c64", "model_type": "refiner", "num_blocks": 16, "channels": 64},
        {"run_name": "refiner_b16_c128", "model_type": "refiner", "num_blocks": 16, "channels": 128},
        {"run_name": "refiner_b16_c192", "model_type": "refiner", "num_blocks": 16, "channels": 192},
        {"run_name": "refiner_b16_c256", "model_type": "refiner", "num_blocks": 16, "channels": 256},
        # {"run_name": "refiner_b16_c320", "model_type": "refiner", "num_blocks": 16, "channels": 320},
        # {"run_name": "refiner_b16_c384", "model_type": "refiner", "num_blocks": 16, "channels": 384},
        # {"run_name": "refiner_b16_c448", "model_type": "refiner", "num_blocks": 16, "channels": 448},
        # {"run_name": "refiner_b16_c512", "model_type": "refiner", "num_blocks": 16, "channels": 512},
        # {"run_name": "refiner_b16_c576", "model_type": "refiner", "num_blocks": 16, "channels": 576},
        # {"run_name": "refiner_b16_c640", "model_type": "refiner", "num_blocks": 16, "channels": 640},
        # {"run_name": "refiner_b16_c712", "model_type": "refiner", "num_blocks": 16, "channels": 712},

        # num_blocks = 24
        {"run_name": "refiner_b24_c64", "model_type": "refiner", "num_blocks": 24, "channels": 64},
        {"run_name": "refiner_b24_c128", "model_type": "refiner", "num_blocks": 24, "channels": 128},
        {"run_name": "refiner_b24_c192", "model_type": "refiner", "num_blocks": 24, "channels": 192},
        {"run_name": "refiner_b24_c256", "model_type": "refiner", "num_blocks": 24, "channels": 256},
        # {"run_name": "refiner_b24_c320", "model_type": "refiner", "num_blocks": 24, "channels": 320},
        # {"run_name": "refiner_b24_c384", "model_type": "refiner", "num_blocks": 24, "channels": 384},
        # {"run_name": "refiner_b24_c448", "model_type": "refiner", "num_blocks": 24, "channels": 448},
        # {"run_name": "refiner_b24_c512", "model_type": "refiner", "num_blocks": 24, "channels": 512},
        # {"run_name": "refiner_b24_c576", "model_type": "refiner", "num_blocks": 24, "channels": 576},
        # {"run_name": "refiner_b24_c640", "model_type": "refiner", "num_blocks": 24, "channels": 640},
        # {"run_name": "refiner_b24_c712", "model_type": "refiner", "num_blocks": 24, "channels": 712},

    ]
    base = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": 75,
        "lr": 1e-4,
        "eta_min": 1e-6,
        "weight_decay": 0.0,
    }

    for cfg in sweep:
        config = {**base, **cfg}
        best_mse = main_run(config)
        print("DONE. Best val MSE:", best_mse)
