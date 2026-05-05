
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from medmnist_dataloader import MedMNISTWrapper, NUM_CLASSES_MAP, TASK_MAP
import os
import time

# ============================================================
# CONFIG
# ============================================================
EXPERIMENT   = "ark_all_12_2d_datasets"
DATASETS     = [
    'PathMNIST',
    'BloodMNIST',
    'DermaMNIST',
    'OCTMNIST',
    'PneumoniaMNIST',
    'RetinaMNIST',
    'BreastMNIST',
    'TissueMNIST',
    'OrganAMNIST',
    'OrganCMNIST',
    'OrganSMNIST',
    'ChestMNIST',
]
IMAGE_SIZE   = 64
BATCH_SIZE   = 8
EPOCHS       = 20
PATIENCE     = 4
LR           = 1e-3
MOMENTUM_EMA = 0.9
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR     = "./outputs/" + EXPERIMENT
os.makedirs(SAVE_DIR, exist_ok=True)
# ============================================================


class ArkMedMNIST(nn.Module):
    def __init__(self, num_classes_list, img_size=64):
        super().__init__()
        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,
            global_pool="avg"
        )
        self.num_features = self.encoder.num_features
        self.omni_heads   = nn.ModuleList([
            nn.Linear(self.num_features, nc)
            for nc in num_classes_list
        ])

    def forward(self, x, head_n):
        features = self.encoder(x)
        return features, self.omni_heads[head_n](features)


def ema_update(student, teacher, momentum):
    with torch.no_grad():
        for s_p, t_p in zip(student.parameters(),
                             teacher.parameters()):
            t_p.data = momentum * t_p.data + \
                       (1 - momentum) * s_p.data


def compute_auc(y_true, y_pred, num_classes, task_type):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    aucs   = []
    if task_type == "multi-class classification":
        y_true_oh = np.zeros((len(y_true), num_classes))
        for i, val in enumerate(y_true):
            y_true_oh[i, int(val)] = 1
        for c in range(num_classes):
            if y_true_oh[:, c].sum() > 0:
                aucs.append(roc_auc_score(
                    y_true_oh[:, c], y_pred[:, c]))
    else:
        for c in range(num_classes):
            if y_true[:, c].sum() > 0:
                aucs.append(roc_auc_score(
                    y_true[:, c], y_pred[:, c]))
    return np.mean(aucs) if aucs else 0.0


def train_one_cycle(model, teacher, dataset_name, loader,
                    head_n, optimizer, epoch, device):
    model.train()
    if TASK_MAP[dataset_name] == "multi-class classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    MSE  = nn.MSELoss()
    coff = min((epoch / EPOCHS) * 0.1, 0.1)

    total_loss = total_cls = total_mse = n = 0

    for v1, v2, labels in loader:
        v1, v2, labels = (v1.to(device),
                          v2.to(device),
                          labels.to(device))

        feat_s, pred_s = model(v1, head_n)
        with torch.no_grad():
            feat_t, _ = teacher(v2, head_n)

        loss_cls   = criterion(pred_s, labels)
        loss_const = MSE(feat_s, feat_t)
        loss       = (1 - coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_mse  += loss_const.item()
        n          += 1

    return total_loss/n, total_cls/n, total_mse/n


def evaluate(model, dataset_name, loader,
             head_n, num_classes, device):
    model.eval()
    all_preds  = []
    all_labels = []
    task_type  = TASK_MAP[dataset_name]

    with torch.no_grad():
        for v1, _, labels in loader:
            v1 = v1.to(device)
            _, pred = model(v1, head_n)
            if task_type == "multi-class classification":
                pred = torch.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return compute_auc(
        all_labels, all_preds, num_classes, task_type)


def main():
    print("=" * 60)
    print("EXPERIMENT : " + EXPERIMENT)
    print("DATASETS   : all 12 MedMNIST 2D datasets")
    print("IMAGE_SIZE : " + str(IMAGE_SIZE))
    print("BATCH_SIZE : " + str(BATCH_SIZE))
    print("EPOCHS     : " + str(EPOCHS))
    print("PATIENCE   : " + str(PATIENCE))
    print("DEVICE     : " + str(DEVICE))
    print("=" * 60)

    # ---- Dataloaders ----
    print("\nBuilding dataloaders...")
    num_classes_list = [NUM_CLASSES_MAP[d] for d in DATASETS]
    train_loaders, val_loaders, test_loaders = [], [], []

    for name in DATASETS:
        train_ds = MedMNISTWrapper(
            name, split="train", size=IMAGE_SIZE)
        val_ds   = MedMNISTWrapper(
            name, split="val",   size=IMAGE_SIZE)
        test_ds  = MedMNISTWrapper(
            name, split="test",  size=IMAGE_SIZE)

        train_loaders.append(DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True,  num_workers=0,
            pin_memory=True))
        val_loaders.append(DataLoader(
            val_ds,   batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0,
            pin_memory=True))
        test_loaders.append(DataLoader(
            test_ds,  batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0,
            pin_memory=True))

        print("  " + name +
              ": train=" + str(len(train_ds)) +
              " val="   + str(len(val_ds))   +
              " test="  + str(len(test_ds)))

    # ---- Models ----
    print("\nBuilding models...")
    model   = ArkMedMNIST(
        num_classes_list, img_size=IMAGE_SIZE).to(DEVICE)
    teacher = ArkMedMNIST(
        num_classes_list, img_size=IMAGE_SIZE).to(DEVICE)

    for p in teacher.parameters():
        p.requires_grad = False

    teacher.load_state_dict(model.state_dict())
    model.encoder.set_grad_checkpointing(True)

    print("  Feature dim : " + str(model.num_features))
    print("  num_classes : " + str(num_classes_list))
    print("  Total params: " + str(
        sum(p.numel() for p in model.parameters()
            if p.requires_grad) // 1_000_000) + "M")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    # ---- Log file ----
    log_path = os.path.join(SAVE_DIR, "train_log.txt")
    with open(log_path, "w") as f:
        f.write("Experiment: " + EXPERIMENT + "\n")
        f.write("Datasets: " + str(DATASETS) + "\n")
        f.write("Image size: " + str(IMAGE_SIZE) + "\n")
        f.write("Batch size: " + str(BATCH_SIZE) + "\n")
        f.write("Epochs: "     + str(EPOCHS)     + "\n")
        f.write("Patience: "   + str(PATIENCE)   + "\n\n")

    # ---- Training ----
    print("\nStarting cyclic pretraining...")
    best_avg_auc  = 0.0
    patience_ctr  = 0
    stopped_epoch = EPOCHS

    for epoch in range(EPOCHS):
        t0 = time.time()
        print("\n" + "=" * 60)
        print("Epoch " + str(epoch+1) + "/" + str(EPOCHS) +
              "   [patience " +
              str(patience_ctr) + "/" + str(PATIENCE) + "]")
        print("=" * 60)

        # Cyclic: one full epoch per dataset
        for i, name in enumerate(DATASETS):
            loss, cls_l, mse_l = train_one_cycle(
                model, teacher, name,
                train_loaders[i], i,
                optimizer, epoch, DEVICE)
            ema_update(model, teacher, MOMENTUM_EMA)
            print("  [" + name + "]" +
                  " loss="  + str(round(loss,  4)) +
                  " cls="   + str(round(cls_l, 4)) +
                  " mse="   + str(round(mse_l, 4)))

        scheduler.step()

        # Validation
        print("\n  Validation AUC:")
        auc_list = []
        for i, name in enumerate(DATASETS):
            auc = evaluate(
                teacher, name,
                val_loaders[i], i,
                num_classes_list[i], DEVICE)
            auc_list.append(auc)
            print("    " + name + ": " + str(round(auc, 4)))

        avg_auc    = np.mean(auc_list)
        epoch_time = time.time() - t0
        print("    Average   : " + str(round(avg_auc, 4)))
        print("  Epoch time  : " + str(round(epoch_time/60, 1)) + " min")

        # Log
        with open(log_path, "a") as f:
            f.write("Epoch " + str(epoch+1) +
                    ": avg_auc=" + str(round(avg_auc, 4)) +
                    " time="    + str(round(epoch_time/60, 1)) + "min\n")
            for name, auc in zip(DATASETS, auc_list):
                f.write("  " + name + ": " +
                        str(round(auc, 4)) + "\n")
            f.write("\n")

        # Early stopping check
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "teacher":    teacher.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "avg_auc":    float(avg_auc),
                "auc_list":   [float(a) for a in auc_list],
                "datasets":   DATASETS,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print("  New best saved: AUC=" +
                  str(round(avg_auc, 4)))
        else:
            patience_ctr += 1
            print("  No improvement. Patience: " +
                  str(patience_ctr) + "/" + str(PATIENCE))
            if patience_ctr >= PATIENCE:
                stopped_epoch = epoch + 1
                print("\n  EARLY STOPPING at epoch " +
                      str(stopped_epoch))
                with open(log_path, "a") as f:
                    f.write("Early stopping at epoch " +
                            str(stopped_epoch) + "\n")
                break

    # ---- Final test ----
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(
        os.path.join(SAVE_DIR, "best_model.pth"),
        weights_only=False)
    teacher.load_state_dict(checkpoint["teacher"])
    print("Best model from epoch " +
          str(checkpoint["epoch"] + 1))

    test_aucs = []
    for i, name in enumerate(DATASETS):
        auc = evaluate(
            teacher, name,
            test_loaders[i], i,
            num_classes_list[i], DEVICE)
        test_aucs.append(auc)
        print("  " + name + " Test AUC: " + str(round(auc, 4)))

    mean_auc = np.mean(test_aucs)
    print("\n  Mean Test AUC : " + str(round(mean_auc, 4)))
    print("  Best Val AUC  : " + str(round(best_avg_auc, 4)))
    print("  Stopped epoch : " + str(stopped_epoch))

    with open(log_path, "a") as f:
        f.write("\nFINAL TEST RESULTS:\n")
        for name, auc in zip(DATASETS, test_aucs):
            f.write("  " + name + ": " +
                    str(round(auc, 4)) + "\n")
        f.write("Mean Test AUC: " + str(round(mean_auc, 4)) + "\n")
        f.write("Stopped epoch: " + str(stopped_epoch) + "\n")

    print("\nResults saved to " + log_path)
    print("=" * 60)


main()
