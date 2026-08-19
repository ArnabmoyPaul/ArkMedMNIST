import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import dict_dataloarder, build_transform_classification
from medmnist_dataloader import MEDMNIST_2D_KEYS, MEDMNIST_DATALOADER_DICT
from models import build_omni_model, load_finetune_backbone, save_checkpoint
from trainer import train_downstream_epoch, evaluate, test_classification
from utils import get_config, metric_AUROC

dict_dataloarder.update(MEDMNIST_DATALOADER_DICT)  # register the 12 MedMNIST 2D classes

DEFAULT_CHECKPOINT = (
    "Models/swin_tiny_medmnist2d_swintiny/"
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist/"
    "medmnist2d_swintiny/"
    "Ark_Plus_pathmnist_bloodmnist_dermamnist_octmnist_pneumoniamnist_retinamnist_"
    "breastmnist_tissuemnist_organamnist_organcmnist_organsmnist_chestmnist24.pth.tar"
)

BENCHMARKS_2D = {
    "chestmnist": 0.773,
    "pathmnist": 0.989,
    "dermamnist": 0.912,
    "octmnist": 0.958,
    "pneumoniamnist": 0.962,
    "retinamnist": 0.716,
    "breastmnist": 0.866,
    "bloodmnist": 0.997,
    "tissuemnist": 0.932,
    "organamnist": 0.998,
    "organcmnist": 0.993,
    "organsmnist": 0.975,
}


def _compare_to_benchmark(auc, target):
    """PASS if auc meets/beats target, WITHIN_1PCT if short by <=0.01,
    FAIL otherwise. NO_TARGET when the benchmark table has no entry."""
    if target is None:
        return "NO_TARGET"
    if auc >= target:
        return "PASS"
    if target - auc <= 0.01 + 1e-9:  # epsilon for float imprecision
        return "WITHIN_1PCT"
    return "FAIL"


def _early_stop(val_losses, patience):
    """True once val_losses hasn't set a new best in the last `patience` epochs."""
    if len(val_losses) <= patience:
        return False
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    return best_idx <= len(val_losses) - 1 - patience


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Downstream fine-tune a 2D MedMNIST omni-pretrained checkpoint")
    p.add_argument("--dataset", required=True, choices=MEDMNIST_2D_KEYS)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_config = get_config("datasets_config_medmnist.yaml")
    ds_cfg = datasets_config[args.dataset]
    diseases = ds_cfg["diseases"]
    multiclass = ds_cfg["task_type"] == "multi-class classification"
    criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()

    crop_size, resize = 112, 128
    train_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["train_list"],
        crop_size=crop_size, resize=resize, augment=None)
    val_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["val_list"],
        crop_size=crop_size, resize=resize,
        augment=build_transform_classification(normalize="imagenet", crop_size=crop_size,
                                                 resize=resize, mode="valid"))
    test_set = dict_dataloarder[args.dataset](
        images_path=ds_cfg["data_dir"], file_path=ds_cfg["test_list"],
        crop_size=crop_size, resize=resize,
        augment=build_transform_classification(normalize="imagenet", crop_size=crop_size,
                                                 resize=resize, mode="test", test_augment=True))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model_args = SimpleNamespace(model_name="swin_tiny", crop_size=crop_size,
                                  projector_features=512, use_mlp=False, pretrained_weights=None)
    model = build_omni_model(model_args, num_classes_list=[len(diseases)])
    load_finetune_backbone(model, args.checkpoint, key="teacher")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model_dir = os.path.join("Models", f"finetune_{args.dataset}")
    output_dir = os.path.join("Outputs", f"finetune_{args.dataset}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "best")

    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_downstream_epoch(model, 0, args.dataset, train_loader, device, criterion,
                                             optimizer, epoch, scaler=scaler)
        val_loss = evaluate(model, 0, val_loader, device, criterion, args.dataset, scaler=scaler)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({(time.time() - epoch_start) / 60:.1f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({"epoch": epoch, "val_loss": val_loss, "state_dict": model.state_dict()},
                             filename=best_path)

        if _early_stop(val_losses, args.patience):
            print(f"Early stopping at epoch {epoch} (no val improvement in {args.patience} epochs)")
            break

    best_ckpt = torch.load(best_path + ".pth.tar", map_location=device, weights_only=False)  # self-generated checkpoint, trusted
    model.load_state_dict(best_ckpt["state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} (val_loss={best_ckpt['val_loss']:.4f})")

    y_test, p_test = test_classification(model, 0, test_loader, device, multiclass, is_3d=False)
    individual_auc = metric_AUROC(y_test, p_test, len(diseases))
    overall_auc = float(np.mean(individual_auc))
    target = BENCHMARKS_2D[args.dataset]
    verdict = _compare_to_benchmark(overall_auc, target)

    results = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "test_auc_per_class": dict(zip(diseases, [float(a) for a in individual_auc])),
        "test_overall_mAUC": overall_auc,
        "benchmark_target": target,
        "verdict": verdict,
    }
    print(f"  Test AUC per class: {results['test_auc_per_class']}")
    print(f"  --- Overall test mAUC: {overall_auc:.4f} (target={target}) -> {verdict} ---")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "results.txt"), "a") as f:
        f.write(json.dumps(results) + "\n")

    return results


if __name__ == "__main__":
    main()
