"""Task 2/3 - Train the baseline U-Net, then the dilated-bottleneck variant, under one protocol.

Usage:
  python train.py                 # full run: baseline then dilated
  python train.py --smoke         # 1-epoch smoke test (separate log / checkpoints)
"""
import argparse
import csv
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import (CLASS_NAMES, IGNORE_INDEX, IMG_SIZE, NUM_CLASSES, PROJECT_DIR,
                     get_dataloaders, write_data_contract)
from evaluate import confusion_matrix, iou_from_cm
from models import SegmentationUNet, block_param_table, count_parameters, format_param_table

CONFIG = {
    "seed": 42,
    "deterministic": {"cudnn.deterministic": True, "cudnn.benchmark": False},
    "resolution": [IMG_SIZE, IMG_SIZE],
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "ignore_index": IGNORE_INDEX,
    "batch_size": 8,
    "max_epochs": 60,
    "early_stopping": {"monitor": "val_mIoU", "mode": "max", "patience": 15,
                       "rule": "stop after 15 consecutive epochs without val-mIoU improvement; keep best-val-mIoU checkpoint"},
    "loss": "CrossEntropyLoss(ignore_index=255), unweighted",
    "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
    "scheduler": {"name": "CosineAnnealingLR", "T_max": 60, "eta_min": 1e-5, "step": "per epoch"},
    "augmentation": "train only: synchronised hflip p=0.5, vflip p=0.5, rot90 k in {1,2,3} p=0.5",
    "data_order": "DataLoader shuffle with torch.Generator seeded 42 and a dedicated random.Random(42) augmentation "
                  "stream, both re-created for each model -> identical batches/augmentations for both models",
    "models": {
        "baseline": "SegmentationUNet(base_ch=32, use_dilated_bottleneck=False): bottleneck = DoubleConv(256,512)",
        "dilated": "SegmentationUNet(base_ch=32, use_dilated_bottleneck=True): bottleneck = ConvBNReLU 3x3 256->512, "
                   "4 parallel ConvBNReLU 3x3 512->128 (dilation=padding=1,2,4,8), concat 512, 1x1 ConvBNReLU 512->512",
    },
    "controlled_change": "only the bottleneck module differs; encoder, decoder, skips, head, data, optimiser, schedule identical",
    "architecture": "4-stage U-Net encoder (32,64,128,256) of DoubleConv [Conv3x3(no bias)+BN+ReLU]x2, MaxPool2x2; "
                    "bottleneck 512ch at 1/16 resolution; decoder ConvTranspose2d 2x2 s2 + concat skip + DoubleConv x4; 1x1 head",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_environment(path):
    import torchvision
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        drv = "n/a"
    lines = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"numpy: {np.__version__}",
        f"CUDA (torch build): {torch.version.cuda}",
        f"cuDNN: {torch.backends.cudnn.version()}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"NVIDIA driver: {drv}",
    ]
    for mod in ("PIL", "matplotlib", "pandas", "kagglehub"):
        try:
            m = __import__(mod)
            lines.append(f"{'Pillow' if mod == 'PIL' else mod}: {m.__version__}")
        except Exception:
            pass
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum, n_valid = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        valid = int((y != IGNORE_INDEX).sum())
        loss_sum += criterion(logits, y).item() * valid  # pixel-weighted mean over the split
        n_valid += valid
        cm += confusion_matrix(logits.argmax(1), y)
    iou = iou_from_cm(cm)
    return loss_sum / max(n_valid, 1), float(np.nanmean(iou))


def train_one(name, dilated, cfg, device, log_path, ckpt_dir, max_epochs):
    set_seed(cfg["seed"])
    train_loader, val_loader, _ = get_dataloaders(batch_size=cfg["batch_size"], seed=cfg["seed"])
    model = SegmentationUNet(num_classes=NUM_CLASSES, base_ch=32, use_dilated_bottleneck=dilated).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["scheduler"]["T_max"], eta_min=cfg["scheduler"]["eta_min"])
    patience = cfg["early_stopping"]["patience"]
    best, best_epoch, since = -1.0, 0, 0
    ckpt = os.path.join(ckpt_dir, f"{name}_best.pt")
    print(f"\n=== {name}: {count_parameters(model):,} params on {device} ===")
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        lr = opt.param_groups[0]["lr"]
        tl, nb = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()
        vl, vmiou = validate(model, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        improved = vmiou > best
        if improved:
            best, best_epoch, since = vmiou, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": vmiou, "name": name,
                        "use_dilated_bottleneck": dilated}, ckpt)
        else:
            since += 1
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([name, epoch, f"{lr:.8f}", f"{tl / nb:.6f}", f"{vl:.6f}", f"{vmiou:.6f}", f"{dt:.3f}"])
        print(f"{name} ep{epoch:3d} lr={lr:.2e} train_loss={tl / nb:.4f} val_loss={vl:.4f} "
              f"val_mIoU={vmiou:.4f} {dt:.1f}s{' *' if improved else ''}")
        if since >= patience:
            print(f"{name}: early stop at epoch {epoch} (best epoch {best_epoch}, val mIoU {best:.4f})")
            break
    return {"best_epoch": best_epoch, "best_val_miou": best, "epochs_run": epoch,
            "stopped_early": since >= patience, "train_time_s": round(time.time() - t_start, 1),
            "params": count_parameters(model), "checkpoint": os.path.relpath(ckpt, PROJECT_DIR)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-epoch run to separate log/checkpoints")
    ap.add_argument("--models", nargs="+", default=["baseline", "dilated"])
    args = ap.parse_args()

    os.chdir(PROJECT_DIR)
    for d in ("logs", "checkpoints", "results"):
        os.makedirs(d, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads(json.dumps(CONFIG))
    max_epochs = 1 if args.smoke else cfg["max_epochs"]
    tag = "smoke_" if args.smoke else ""
    log_path = os.path.join("logs", f"{tag}training_log.csv")
    ckpt_dir = os.path.join("checkpoints", "smoke") if args.smoke else "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    write_environment("environment.txt")
    write_data_contract()
    base, var = SegmentationUNet(use_dilated_bottleneck=False), SegmentationUNet(use_dilated_bottleneck=True)
    rows = block_param_table(base, var)
    table = format_param_table(rows)
    print(table)
    with open(os.path.join("results", "param_table.txt"), "w") as f:
        f.write(table + "\n")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["model", "epoch", "lr", "train_loss", "val_loss", "val_miou", "epoch_time_s"])
    summary = {}
    for name in args.models:
        summary[name] = train_one(name, name == "dilated", cfg, device, log_path, ckpt_dir, max_epochs)

    if not args.smoke:
        cfg["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cfg["parameter_table"] = [{"block": b, "baseline": pb, "dilated": pv, "delta": d} for b, pb, pv, d in rows]
        cfg["training_outcome"] = summary
        with open("experiment_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
    with open(os.path.join("logs", f"{tag}training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
