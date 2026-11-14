"""Task 4 - Exact evaluation script: test-split metrics, latency, figures, dashboard.

Usage: python evaluate.py   (after train.py has produced checkpoints/ and logs/training_log.csv)

The test split is run exactly once per model. Ground truth = the human-annotated masks.
All numbers in figures/tables are computed here from the checkpoints and the training log.
"""
import csv
import json
import os
import platform
import time

import numpy as np
import torch

NUM_CLASSES_DEFAULT = 5
IGNORE = 255


# ----------------------------------------------------------------------------- metrics
def confusion_matrix(pred, target, num_classes=NUM_CLASSES_DEFAULT, ignore_index=IGNORE):
    """Rows = ground truth, cols = prediction. Pixels with target == ignore_index are excluded."""
    pred, target = pred.reshape(-1), target.reshape(-1)
    keep = target != ignore_index
    idx = target[keep] * num_classes + pred[keep]
    cm = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm.cpu().numpy().astype(np.int64)


def iou_from_cm(cm):
    """Per-class IoU = TP / (TP + FP + FN); NaN where the class is absent from GT and prediction."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


# ----------------------------------------------------------------------------- latency
@torch.no_grad()
def measure_latency(model, device, warmup=50, runs=200, size=256):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(x)
    sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def latency_stats(times_per_repeat):
    """Headline = median of the per-repeat medians; mean/sd/p95 over all timed passes pooled."""
    allt = np.concatenate(times_per_repeat)
    meds = [float(np.median(t)) for t in times_per_repeat]
    return {"median_ms": float(np.median(meds)), "per_repeat_median_ms": meds,
            "mean_ms": float(allt.mean()), "std_ms": float(allt.std()), "p95_ms": float(np.percentile(allt, 95)),
            "min_ms": float(allt.min()), "max_ms": float(allt.max()), "n_timed_passes": int(allt.size)}


# ----------------------------------------------------------------------------- plotting
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"baseline": "#2a78d6", "dilated": "#eb6834"}  # categorical slots 1, 2


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def colorize(mask, class_colors, unlabeled):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[:] = unlabeled
    for c, col in enumerate(class_colors):
        rgb[mask == c] = col
    return rgb


def error_map(pred, gt):
    out = np.full(gt.shape + (3,), 128, dtype=np.uint8)  # grey = ignored (255) pixels
    valid = gt != IGNORE
    out[valid & (pred == gt)] = (0, 170, 0)
    out[valid & (pred != gt)] = (220, 30, 30)
    return out


def denorm(x, mean, std):
    img = x.cpu().numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def image_grid(plt, rows, path, title, class_names, class_colors, unlabeled):
    from matplotlib.patches import Patch
    cols = ["Source image", "Reference mask (human)", "Baseline prediction", "Dilated prediction",
            "Error map (dilated)"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 3.6 * len(rows) + 0.9), squeeze=False)
    for r, row in enumerate(rows):
        panels = [row["img"], colorize(row["gt"], class_colors, unlabeled), colorize(row["pb"], class_colors, unlabeled),
                  colorize(row["pv"], class_colors, unlabeled), error_map(row["pv"], row["gt"])]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=9.5)
        axes[r, 0].set_ylabel(row["name"], fontsize=10, color=INK)
        axes[r, 2].set_xlabel(f"mIoU {row['mb']:.3f}", fontsize=9)
        axes[r, 3].set_xlabel(f"mIoU {row['mv']:.3f} ({row['mv'] - row['mb']:+.3f})", fontsize=9)
    handles = [Patch(color=np.array(col) / 255, label=n) for n, col in zip(class_names, class_colors)]
    handles.append(Patch(color=np.array(unlabeled) / 255, label="Unlabeled / ignored (255)"))
    handles += [Patch(color=(0, 170 / 255, 0), label="Error map: correct"), Patch(color=(220 / 255, 30 / 255, 30 / 255), label="Error map: wrong"),
                Patch(color=(0.5, 0.5, 0.5), label="Error map: ignored")]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8.5)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def read_log(path):
    log = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            log.setdefault(r["model"], []).append({k: (float(v) if k != "model" else v) for k, v in r.items()})
    return log


def plot_training_curves(plt, log, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for key, ax, title in [("train_loss", axes[0], "Training loss (mean over batches)"),
                           ("val_loss", axes[1], "Validation loss (pixel-weighted)"),
                           ("val_miou", axes[2], "Validation mIoU")]:
        for m, rows in log.items():
            ep = [r["epoch"] for r in rows]
            ax.plot(ep, [r[key] for r in rows], color=MODEL_COLORS.get(m), lw=2, label=m)
            if key == "val_miou":
                b = int(np.argmax([r[key] for r in rows]))
                ax.plot(ep[b], rows[b][key], "o", ms=8, color=MODEL_COLORS.get(m), mec=SURFACE, mew=2)
                ax.annotate(f"best {rows[b][key]:.3f} @ep{int(ep[b])}", (ep[b], rows[b][key]),
                            textcoords="offset points", xytext=(4, 6 if m == "baseline" else -14), fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left"); ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_dashboard(plt, summary, log, path, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    b_iou = np.array([summary["models"]["baseline"]["per_class_iou"][c] for c in class_names], dtype=float)
    v_iou = np.array([summary["models"]["dilated"]["per_class_iou"][c] for c in class_names], dtype=float)
    x = np.arange(len(class_names))
    w = 0.38
    ax = axes[0, 0]
    for off, vals, m in [(-w / 2 - 0.01, b_iou, "baseline"), (w / 2 + 0.01, v_iou, "dilated")]:
        bars = ax.bar(x + off, np.nan_to_num(vals), w, color=MODEL_COLORS[m], label=m)
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, (0 if np.isnan(v) else v) + 0.01,
                    "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, class_names); ax.set_ylim(0, 1.05); ax.set_ylabel("IoU")
    ax.set_title("Test per-class IoU", loc="left"); ax.legend(loc="upper right")

    ax = axes[0, 1]
    d = v_iou - b_iou
    ax.bar(x, np.nan_to_num(d), 0.55, color=[MODEL_COLORS["dilated"] if v >= 0 else MODEL_COLORS["baseline"] for v in np.nan_to_num(d)])
    for xi, v in zip(x, d):
        ax.text(xi, np.nan_to_num(v) + (0.003 if np.nan_to_num(v) >= 0 else -0.003), "n/a" if np.isnan(v) else f"{v:+.3f}",
                ha="center", va="bottom" if np.nan_to_num(v) >= 0 else "top", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, class_names); ax.set_ylabel("IoU(dilated) - IoU(baseline)")
    lim = max(0.02, np.nanmax(np.abs(d)) * 1.35)
    ax.set_ylim(-lim, lim)
    ax.set_title("Per-class change (orange = dilated better, blue = baseline better)", loc="left")

    ax = axes[1, 0]
    for m, rows in log.items():
        ax.plot([r["epoch"] for r in rows], [r["val_miou"] for r in rows], color=MODEL_COLORS.get(m), lw=2, label=m)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mIoU"); ax.set_title("Validation mIoU (from training_log.csv)", loc="left")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    mb, mv = summary["models"]["baseline"], summary["models"]["dilated"]
    rows = [
        ["Test mIoU", f"{mb['miou']:.4f}", f"{mv['miou']:.4f}", f"{mv['miou'] - mb['miou']:+.4f}"],
        ["Pixel accuracy", f"{mb['pixel_accuracy']:.4f}", f"{mv['pixel_accuracy']:.4f}", f"{mv['pixel_accuracy'] - mb['pixel_accuracy']:+.4f}"],
        ["Parameters", f"{mb['params']:,}", f"{mv['params']:,}", f"{mv['params'] - mb['params']:+,}"],
        ["Conv MACs (G) @256", f"{mb['conv_macs_256'] / 1e9:.3f}", f"{mv['conv_macs_256'] / 1e9:.3f}",
         f"{(mv['conv_macs_256'] - mb['conv_macs_256']) / 1e6:+.1f} M"],
        ["Latency median* (ms)", f"{mb['latency']['median_ms']:.2f}", f"{mv['latency']['median_ms']:.2f}",
         f"{mv['latency']['median_ms'] - mb['latency']['median_ms']:+.2f}"],
        ["Latency mean±sd (ms)", f"{mb['latency']['mean_ms']:.2f}±{mb['latency']['std_ms']:.2f}",
         f"{mv['latency']['mean_ms']:.2f}±{mv['latency']['std_ms']:.2f}", ""],
        ["Best val mIoU (epoch)", f"{mb['best_val_miou']:.4f} ({mb['best_epoch']})", f"{mv['best_val_miou']:.4f} ({mv['best_epoch']})", ""],
        ["Epochs run", str(mb["epochs_run"]), str(mv["epochs_run"]), ""],
    ]
    t = ax.table(cellText=rows, colLabels=["", "baseline", "dilated", "delta"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold")
    lc = summary["latency_conditions"]
    ax.set_title(f"Summary - test n={summary['n_test_images']} images\nLatency device: {lc['device']}\nbatch {lc['batch_size']}, "
                 f"{lc['dtype']}, {lc['repeats']} x ({lc['warmup']} warm-up + {lc['timed_runs']} timed passes)", loc="left", fontsize=9.5)
    fig.suptitle("Baseline U-Net vs dilated bottleneck - Dubai aerial imagery (single seed, single split)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dataset import (CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX, MEAN, NUM_CLASSES, PROJECT_DIR, STD,
                         UNLABELED_COLOR, get_dataloaders)
    from models import SegmentationUNet, count_conv_macs, count_parameters
    from train import set_seed

    os.chdir(PROJECT_DIR)
    os.makedirs("results", exist_ok=True)
    _style(plt)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=8, seed=42)
    test_names = test_loader.dataset.names
    train_summary = json.load(open(os.path.join("logs", "training_summary.json")))

    # ---- GT pixel counts on the test set
    gt_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_ignored = 0
    for _, y, _ in test_loader:
        yv = y[y != IGNORE_INDEX]
        gt_counts += np.bincount(yv.numpy(), minlength=NUM_CLASSES)
        n_ignored += int((y == IGNORE_INDEX).sum())
    print("Test-set GT pixel counts (256x256):")
    for c, n in zip(CLASS_NAMES, gt_counts):
        print(f"  {c:<11}{n:>10,}  ({n / gt_counts.sum():6.2%})")
    print(f"  {'ignored':<11}{n_ignored:>10,}")

    models, per_image, preds, nets = {}, {}, {}, {}
    LAT_REPEATS = 5
    for name in ("baseline", "dilated"):
        ck = torch.load(os.path.join("checkpoints", f"{name}_best.pt"), map_location=device, weights_only=True)
        model = SegmentationUNet(num_classes=NUM_CLASSES, base_ch=32, use_dilated_bottleneck=ck["use_dilated_bottleneck"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        per_image[name], preds[name] = {}, {}
        with torch.no_grad():  # single pass over the test split
            for x, y, i in test_loader:
                p = model(x.to(device)).argmax(1).cpu()
                c = confusion_matrix(p, y, NUM_CLASSES, IGNORE_INDEX)
                cm += c
                n = test_names[int(i)]
                per_image[name][n] = float(np.nanmean(iou_from_cm(c)))
                preds[name][n] = p[0].numpy()
        iou = iou_from_cm(cm)
        nets[name] = model
        models[name] = {
            "checkpoint_epoch": ck["epoch"],
            "per_class_iou": {c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASS_NAMES, iou)},
            "miou": float(np.nanmean(iou)),
            "pixel_accuracy": float(np.trace(cm) / cm.sum()),
            "params": count_parameters(model),
            "conv_macs_256": count_conv_macs(model),
            "confusion_matrix": cm.tolist(),
            "per_image_miou": per_image[name],
            "best_val_miou": train_summary[name]["best_val_miou"],
            "best_epoch": train_summary[name]["best_epoch"],
            "epochs_run": train_summary[name]["epochs_run"],
        }
        print(f"{name}: test mIoU={models[name]['miou']:.4f} params={models[name]['params']:,}")

    # ---- latency: the 50 warm-up + 200 timed protocol, repeated LAT_REPEATS times, models interleaved
    times = {n: [] for n in nets}
    for r in range(LAT_REPEATS):
        for n in nets:
            times[n].append(measure_latency(nets[n], device))
    for n in nets:
        models[n]["latency"] = latency_stats(times[n])
        lat = models[n]["latency"]
        print(f"{n}: latency median-of-medians={lat['median_ms']:.3f} ms; per-repeat medians="
              f"{[round(v, 2) for v in lat['per_repeat_median_ms']]}; pooled mean {lat['mean_ms']:.3f} +/- {lat['std_ms']:.3f} ms")

    summary = {
        "n_test_images": len(test_names),
        "test_images": test_names,
        "class_names": CLASS_NAMES,
        "test_gt_pixel_counts": {**{c: int(n) for c, n in zip(CLASS_NAMES, gt_counts)}, "ignored_255": n_ignored},
        "metric_definition": "IoU per class from one confusion matrix accumulated over all test pixels (GT==255 excluded); "
                             "mIoU = mean over classes with non-empty union; evaluated at 256x256 resized resolution; "
                             "confusion_matrix rows=GT, cols=prediction",
        "latency_conditions": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
            "device_type": device.type, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "batch_size": 1, "input_shape": [1, 3, 256, 256], "dtype": "FP32", "mode": "model.eval() + torch.no_grad()",
            "warmup": 50, "timed_runs": 200, "repeats": LAT_REPEATS,
            "repeat_scheme": "each repeat = fresh 50 warm-up + 200 timed passes; models alternated baseline/dilated per repeat; "
                             "headline latency = median of the per-repeat medians",
            "timing": "time.perf_counter around each forward pass, torch.cuda.synchronize() after each pass; "
                      "forward pass only (no data loading, pre/post-processing or argmax)",
        },
        "models": models,
    }
    with open(os.path.join("results", "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "per_class_iou.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "test_gt_pixels", "iou_baseline", "iou_dilated", "delta"])
        for c, n in zip(CLASS_NAMES, gt_counts):
            b, v = models["baseline"]["per_class_iou"][c], models["dilated"]["per_class_iou"][c]
            w.writerow([c, int(n), b, v, None if b is None or v is None else v - b])
        w.writerow(["mIoU", int(gt_counts.sum()), models["baseline"]["miou"], models["dilated"]["miou"],
                    models["dilated"]["miou"] - models["baseline"]["miou"]])
    with open(os.path.join("results", "per_image_miou.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "miou_baseline", "miou_dilated", "gain"])
        for n in test_names:
            w.writerow([n, per_image["baseline"][n], per_image["dilated"][n], per_image["dilated"][n] - per_image["baseline"][n]])

    # ---- qualitative figures
    imgs = {test_names[int(i)]: (denorm(x[0], MEAN, STD), y[0].numpy()) for x, y, i in test_loader}

    def row(n):
        return {"name": n, "img": imgs[n][0], "gt": imgs[n][1], "pb": preds["baseline"][n], "pv": preds["dilated"][n],
                "mb": per_image["baseline"][n], "mv": per_image["dilated"][n]}

    gain = sorted(test_names, key=lambda n: per_image["dilated"][n] - per_image["baseline"][n], reverse=True)[:3]
    worst = sorted(test_names, key=lambda n: (per_image["baseline"][n] + per_image["dilated"][n]) / 2)[:3]
    image_grid(plt, [row(n) for n in gain], os.path.join("results", "qualitative_comparison.png"),
               "Top-3 test images by per-image mIoU gain (dilated - baseline)", CLASS_NAMES, CLASS_COLORS, UNLABELED_COLOR)
    image_grid(plt, [row(n) for n in worst], os.path.join("results", "failure_cases.png"),
               "Failure cases: 3 lowest test images by per-image mIoU (mean of both models)", CLASS_NAMES, CLASS_COLORS, UNLABELED_COLOR)

    log = read_log(os.path.join("logs", "training_log.csv"))
    plot_training_curves(plt, log, os.path.join("results", "training_curves.png"))
    plot_dashboard(plt, summary, log, os.path.join("results", "results_dashboard.png"), CLASS_NAMES)
    print("qualitative (gain):", gain)
    print("failure cases:", worst)
    print("saved results/metrics_summary.json, per_class_iou.csv, per_image_miou.csv and figures")


if __name__ == "__main__":
    main()
