"""Task 1 - Data contract for the Dubai aerial imagery segmentation experiment.

Dataset : "Semantic segmentation of aerial imagery" (Humans in the Loop / MBRSC, Dubai), CC0.
Source  : kagglehub.dataset_download("humansintheloop/semantic-segmentation-of-aerial-imagery")
Masks   : human-annotated RGB colour masks (NOT algorithm-generated pseudo-ground truth).

Run `python dataset.py` to (down)load the data, build the split and write data_contract.json.
"""
import json
import os
import random
import re
import shutil

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_ID = "humansintheloop/semantic-segmentation-of-aerial-imagery"
DATASET_DIRNAME = "Semantic segmentation dataset"

# ---------------------------------------------------------------- class contract
CLASS_NAMES = ["Building", "Land", "Road", "Vegetation", "Water"]
CLASS_COLORS = [  # RGB values actually used in the mask PNGs
    (60, 16, 152),    # 0 Building
    (132, 41, 246),   # 1 Land (unpaved area)
    (110, 193, 228),  # 2 Road
    (254, 221, 58),   # 3 Vegetation
    (226, 169, 41),   # 4 Water
]
UNLABELED_COLOR = (155, 155, 155)
NUM_CLASSES = 5
IGNORE_INDEX = 255
MAX_L1_DIST = 45  # nearest-palette L1 distance >= this -> ignore

# ---------------------------------------------------------------- transform contract
IMG_SIZE = 256
MEAN = (0.485, 0.456, 0.406)  # ImageNet
STD = (0.229, 0.224, 0.225)
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Palette used for nearest-colour lookup: 5 classes + Unlabeled (index 5 -> ignore)
_PALETTE = np.array(CLASS_COLORS + [UNLABELED_COLOR], dtype=np.int32)  # (6,3)


def get_data_root(data_dir=None):
    """Return the directory that contains 'Tile 1'..'Tile 8'.

    Uses <project>/data/<DATASET_DIRNAME> if present, otherwise downloads with kagglehub
    and copies it there (data/ is excluded from the submission zip).
    """
    data_dir = data_dir or os.path.join(PROJECT_DIR, "data")
    local = os.path.join(data_dir, DATASET_DIRNAME)
    if not os.path.isdir(local):
        import kagglehub
        dl = kagglehub.dataset_download(KAGGLE_ID)
        src = os.path.join(dl, DATASET_DIRNAME)
        if not os.path.isdir(src):  # adapt if the archive layout differs
            cands = [r for r, d, _ in os.walk(dl) if any(x.lower().startswith("tile") for x in d)]
            if not cands:
                raise FileNotFoundError(f"No 'Tile *' folders found under {dl}")
            src = cands[0]
        os.makedirs(data_dir, exist_ok=True)
        shutil.copytree(src, local)
    return local


def list_pairs(root):
    """Pair image/mask files *within each tile* (file names repeat across tiles).

    Returns a deterministically sorted list of dicts {name, image, mask} with paths
    relative to root. Sample name e.g. 'Tile3_part_005'.
    """
    pairs = []
    tiles = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.lower().startswith("tile")],
        key=lambda s: int(re.findall(r"\d+", s)[0]),
    )
    for tile in tiles:
        tnum = int(re.findall(r"\d+", tile)[0])
        img_dir, msk_dir = os.path.join(root, tile, "images"), os.path.join(root, tile, "masks")
        masks = {os.path.splitext(f)[0]: f for f in os.listdir(msk_dir) if f.lower().endswith(".png")}
        for f in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in (".jpg", ".jpeg", ".png") or stem not in masks:
                continue
            part = re.findall(r"\d+", stem)[-1]
            pairs.append({
                "name": f"Tile{tnum}_part_{part}",
                "image": f"{tile}/images/{f}",
                "mask": f"{tile}/masks/{masks[stem]}",
            })
    return pairs


def make_splits(pairs, seed=SPLIT_SEED):
    """Shuffle all pairs with random.Random(seed) and cut 70/15/15 (72 -> 50/10/12)."""
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(SPLIT_FRACTIONS[0] * n)
    n_val = int(SPLIT_FRACTIONS[1] * n)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def rgb_mask_to_index(rgb):
    """(H,W,3) uint8 RGB -> (H,W) uint8 class index via nearest palette colour (L1).

    Nearest = Unlabeled, or L1 distance >= MAX_L1_DIST -> IGNORE_INDEX (255).
    """
    flat = rgb.reshape(-1, 3).astype(np.int32)
    out = np.empty(flat.shape[0], dtype=np.uint8)
    for s in range(0, flat.shape[0], 1 << 20):  # chunk to bound memory
        d = np.abs(flat[s:s + (1 << 20), None, :] - _PALETTE[None]).sum(-1)  # (N,6)
        idx = d.argmin(1)
        dist = d[np.arange(len(idx)), idx]
        lab = idx.astype(np.uint8)
        lab[(idx == len(CLASS_COLORS)) | (dist >= MAX_L1_DIST)] = IGNORE_INDEX
        out[s:s + (1 << 20)] = lab
    return out.reshape(rgb.shape[:2])


def load_sample(root, pair, size=IMG_SIZE):
    """Deterministic part of the transform: resize + to_tensor + normalise / index mapping."""
    img = Image.open(os.path.join(root, pair["image"])).convert("RGB")
    msk = Image.open(os.path.join(root, pair["mask"])).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    # Nearest resize of the RGB mask picks original pixels only, so mapping after resizing
    # is identical to mapping first and nearest-resizing the label map.
    msk = msk.resize((size, size), Image.NEAREST)
    x = TF.normalize(TF.to_tensor(img), MEAN, STD)
    y = torch.from_numpy(rgb_mask_to_index(np.array(msk)).astype(np.int64))
    return x, y


class DubaiSegDataset(Dataset):
    """Pre-loads and caches the resized samples (72 images -> tiny)."""

    def __init__(self, root, pairs, train=False, aug_seed=SPLIT_SEED):
        self.root, self.pairs, self.train = root, pairs, train
        self.names = [p["name"] for p in pairs]
        self.cache = [load_sample(root, p) for p in pairs]
        self.rng = random.Random(aug_seed)  # dedicated RNG -> identical aug stream per model

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        x, y = self.cache[i]
        if self.train:  # synchronised geometric augmentation (exact, no interpolation)
            if self.rng.random() < 0.5:
                x, y = torch.flip(x, [2]), torch.flip(y, [1])
            if self.rng.random() < 0.5:
                x, y = torch.flip(x, [1]), torch.flip(y, [0])
            if self.rng.random() < 0.5:
                k = self.rng.choice([1, 2, 3])  # 90/180/270 degrees
                x, y = torch.rot90(x, k, [1, 2]), torch.rot90(y, k, [0, 1])
        return x.contiguous(), y.contiguous(), i


def get_dataloaders(batch_size=8, seed=SPLIT_SEED, data_dir=None):
    root = get_data_root(data_dir)
    splits = make_splits(list_pairs(root), seed)
    g = torch.Generator().manual_seed(seed)  # fixed batch order, independent of model init
    train_ds = DubaiSegDataset(root, splits["train"], train=True, aug_seed=seed)
    val_ds = DubaiSegDataset(root, splits["val"])
    test_ds = DubaiSegDataset(root, splits["test"])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0),
    )


def split_pixel_counts(root, split_pairs):
    counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)  # last = ignored
    for p in split_pairs:
        _, y = load_sample(root, p)
        y = y.numpy()
        counts[:NUM_CLASSES] += np.bincount(y[y != IGNORE_INDEX].ravel(), minlength=NUM_CLASSES)
        counts[NUM_CLASSES] += int((y == IGNORE_INDEX).sum())
    return counts


def write_data_contract(path=None, data_dir=None):
    path = path or os.path.join(PROJECT_DIR, "data_contract.json")
    root = get_data_root(data_dir)
    pairs = list_pairs(root)
    splits = make_splits(pairs)
    pix = {}
    for k, v in splits.items():
        c = split_pixel_counts(root, v)
        pix[k] = {**{CLASS_NAMES[i]: int(c[i]) for i in range(NUM_CLASSES)}, "ignored_255": int(c[-1])}
    contract = {
        "dataset": "Semantic segmentation of aerial imagery (Humans in the Loop / MBRSC, Dubai), CC0",
        "source": f"kagglehub.dataset_download('{KAGGLE_ID}')",
        "annotation": "Human-annotated RGB masks (no algorithm-generated pseudo ground truth used)",
        "num_pairs": len(pairs),
        "pairing": "image_part_XXX.jpg <-> image_part_XXX.png within each tile; sample name TileN_part_XXX",
        "classes": [{"id": i, "name": CLASS_NAMES[i], "rgb": list(CLASS_COLORS[i])} for i in range(NUM_CLASSES)],
        "ignore": {
            "index": IGNORE_INDEX,
            "unlabeled_rgb": list(UNLABELED_COLOR),
            "rule": f"nearest palette colour by L1; if nearest is Unlabeled or L1 distance >= {MAX_L1_DIST} -> {IGNORE_INDEX}",
            "loss_and_metrics": "ignored in CrossEntropyLoss(ignore_index=255) and excluded from confusion matrices",
        },
        "note_classes_json": "The dataset's classes.json lists different hex colours from those in the PNG masks; "
                             "the palette above is the one actually present in the masks.",
        "observed_mask_colours": "Inspection of all 72 masks found exactly 7 RGB values: the 5 class colours, Unlabeled "
                                 "(155,155,155) and black (0,0,0) (~0.5% of pixels, tile borders). Black is >=228 L1 from "
                                 "every palette colour and is therefore mapped to 255 by the distance rule. 18 masks are "
                                 "palette-mode (P) PNGs and are converted to RGB before mapping.",
        "split": {
            "method": f"all pairs sorted by (tile, file), shuffled with random.Random({SPLIT_SEED}), cut 70/15/15",
            "sizes": {k: len(v) for k, v in splits.items()},
            "leakage_warning": "split is random across tiles: train/val/test images come from the same tiles",
            "files": {k: [{"name": p["name"], "image": p["image"], "mask": p["mask"]} for p in v] for k, v in splits.items()},
        },
        "pixel_counts_256": pix,
        "transforms": {
            "image": f"PIL bilinear resize to {IMG_SIZE}x{IMG_SIZE} -> to_tensor -> Normalize(mean={MEAN}, std={STD})",
            "mask": f"PIL nearest resize to {IMG_SIZE}x{IMG_SIZE} -> nearest-palette index mapping",
            "train_augmentation": "synchronised image+mask: hflip p=0.5, vflip p=0.5, rot90 by k in {1,2,3} (90/180/270) p=0.5; "
                                  "rot90/flip are exact pixel permutations (equivalent to nearest for masks)",
            "val_test_augmentation": "none",
        },
    }
    with open(path, "w") as f:
        json.dump(contract, f, indent=2)
    return contract


if __name__ == "__main__":
    c = write_data_contract()
    print("split sizes:", c["split"]["sizes"])
    for k, v in c["pixel_counts_256"].items():
        print(k, v)
