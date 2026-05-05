
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from medmnist3d_dataloader import (
    MedMNIST3DWrapper, NUM_CLASSES_MAP_3D, TASK_MAP_3D
)
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
EXPERIMENT   = 'ark_3d_3datasets'
DATASETS     = ['OrganMNIST3D', 'NoduleMNIST3D', 'FractureMNIST3D']
BATCH_SIZE   = 16
EPOCHS       = 200
LR           = 1e-3
MOMENTUM_EMA = 0.9
FEAT_DIM     = 128
USE_AMP      = True
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR     = os.path.join('./outputs', EXPERIMENT)
os.makedirs(SAVE_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


# ─── 3D BACKBONE ──────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class Ark3D(nn.Module):
    """
    Ark+ teacher-student model for 3D medical volumes.

    forward(x, head_n) → (feat, pred)
        feat : (B, FEAT_DIM)    — drives EMA consistency loss
        pred : (B, num_classes) — drives classification loss

    Training loss (per Ark+ paper):
        loss = (1 - coff) * CE(pred_s, y) + coff * MSE(feat_s, feat_t)
    """

    def __init__(self, num_classes_list, feat_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResBlock3D(32,  64,  stride=2),
            ResBlock3D(64,  128, stride=2),
            ResBlock3D(128, 256, stride=2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.omni_heads = nn.ModuleList(
            [nn.Linear(256, nc) for nc in num_classes_list]
        )

    def forward(self, x, head_n):
        enc  = self.encoder(x)              # (B, 256)
        feat = self.projector(enc)          # (B, feat_dim)
        pred = self.omni_heads[head_n](enc) # (B, num_classes)
        return feat, pred


# ─── UTILITIES ────────────────────────────────────────────────────────────────
def ema_update(student, teacher, momentum):
    with torch.no_grad():
        for s_p, t_p in zip(student.parameters(), teacher.parameters()):
            t_p.data = momentum * t_p.data + (1.0 - momentum) * s_p.data


def compute_auc(y_true, y_pred, num_classes):
    y_true = y_true.numpy().astype(int)
    y_pred = y_pred.numpy()
    y_true_oh = np.eye(num_classes)[y_true]
    aucs = []
    for c in range(num_classes):
        if y_true_oh[:, c].sum() > 0:
            aucs.append(roc_auc_score(y_true_oh[:, c], y_pred[:, c]))
    return float(np.mean(aucs)) if aucs else 0.5


# ─── TRAIN ONE EPOCH FOR ONE DATASET ──────────────────────────────────────────
def train_one_cycle(model, teacher, loader, head_n,
                    optimizer, scaler, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()
    coff = min((epoch / EPOCHS) * 0.1, 0.1)  # ramps 0 → 0.1

    total_loss = total_cls = total_mse = n = 0

    for v1, v2, labels in loader:
        v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            feat_s, pred_s = model(v1, head_n)
            with torch.no_grad():
                feat_t, _ = teacher(v2, head_n)
            loss_cls   = criterion(pred_s, labels)
            loss_const = mse_loss(feat_s, feat_t)
            loss       = (1.0 - coff) * loss_cls + coff * loss_const

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_mse  += loss_const.item()
        n += 1

    return total_loss / n, total_cls / n, total_mse / n


# ─── VALIDATION ───────────────────────────────────────────────────────────────
def evaluate(model, loader, head_n, num_classes, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for v1, _, labels in loader:
            v1 = v1.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _, pred = model(v1, head_n)
            all_preds.append(torch.softmax(pred, dim=1).cpu())
            all_labels.append(labels.cpu())
    return compute_auc(
        torch.cat(all_labels), torch.cat(all_preds), num_classes
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print(f'EXPERIMENT : {EXPERIMENT}')
    print(f'DATASETS   : {DATASETS}')
    print(f'BATCH_SIZE : {BATCH_SIZE}')
    print(f'EPOCHS     : {EPOCHS}')
    print(f'LR         : {LR}')
    print(f'EMA mom    : {MOMENTUM_EMA}')
    print(f'DEVICE     : {DEVICE}')
    print(f'AMP        : {USE_AMP}')
    print('=' * 60)

    # ── Dataloaders ────────────────────────────────────────────────────────────
    num_classes_list = [NUM_CLASSES_MAP_3D[d] for d in DATASETS]
    train_loaders, val_loaders, test_loaders = [], [], []

    print('\nDownloading & building 3D dataloaders...')
    for name in DATASETS:
        tr = MedMNIST3DWrapper(name, split='train')
        va = MedMNIST3DWrapper(name, split='val')
        te = MedMNIST3DWrapper(name, split='test')
        train_loaders.append(DataLoader(tr, batch_size=BATCH_SIZE,
                                        shuffle=True,  num_workers=0, pin_memory=True))
        val_loaders  .append(DataLoader(va, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0, pin_memory=True))
        test_loaders .append(DataLoader(te, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0, pin_memory=True))
        print(f'  {name:20s}  train={len(tr):5d}  val={len(va):5d}  test={len(te):5d}'
              f'  classes={NUM_CLASSES_MAP_3D[name]}')

    # ── Models ─────────────────────────────────────────────────────────────────
    print('\nBuilding Ark3D student + teacher...')
    model   = Ark3D(num_classes_list, feat_dim=FEAT_DIM).to(DEVICE)
    teacher = Ark3D(num_classes_list, feat_dim=FEAT_DIM).to(DEVICE)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params : {n_params / 1e6:.2f} M')
    print(f'  Head sizes       : {num_classes_list}')

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_path = os.path.join(SAVE_DIR, 'train_log.txt')
    history  = {
        'epoch':       [],
        'avg_auc':     [],
        'per_dataset': {d: [] for d in DATASETS},
    }
    with open(log_path, 'w') as f:
        f.write(f'Experiment : {EXPERIMENT}\n')
        f.write(f'Datasets   : {DATASETS}\n')
        f.write(f'Batch size : {BATCH_SIZE}  Epochs : {EPOCHS}\n\n')

    # ── Training loop ──────────────────────────────────────────────────────────
    best_avg_auc = 0.0
    print('\nStarting Ark+ 3D cyclic training...\n')

    for epoch in range(EPOCHS):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f"{'='*60}")

        # One full pass through each dataset (cyclic, mirrors Ark+)
        for i, name in enumerate(DATASETS):
            loss, cls_l, mse_l = train_one_cycle(
                model, teacher, train_loaders[i], i,
                optimizer, scaler, epoch, DEVICE
            )
            ema_update(model, teacher, MOMENTUM_EMA)
            print(f'  [{name:20s}]  '
                  f'loss={loss:.4f}  cls={cls_l:.4f}  mse={mse_l:.4f}')

        scheduler.step()

        # Validation (always run on teacher)
        print('\n  Validation AUC (teacher):')
        auc_list = []
        for i, name in enumerate(DATASETS):
            auc = evaluate(
                teacher, val_loaders[i], i, num_classes_list[i], DEVICE
            )
            auc_list.append(auc)
            history['per_dataset'][name].append(auc)
            print(f'    {name:20s}: {auc:.4f}')

        avg_auc    = float(np.mean(auc_list))
        epoch_secs = time.time() - t0
        print(f"    {'Average':20s}: {avg_auc:.4f}")
        print(f'  Epoch time : {epoch_secs / 60:.1f} min')

        history['epoch'].append(epoch + 1)
        history['avg_auc'].append(avg_auc)

        with open(log_path, 'a') as f:
            f.write(f'Epoch {epoch+1}: avg_auc={avg_auc:.4f}  '
                    f'time={epoch_secs/60:.1f}min\n')
            for name, auc in zip(DATASETS, auc_list):
                f.write(f'  {name}: {auc:.4f}\n')
            f.write('\n')

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'teacher':    teacher.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'avg_auc':    avg_auc,
                'auc_list':   auc_list,
                'datasets':   DATASETS,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f'  ★ New best saved  AUC={avg_auc:.4f}')

    # ── Final test ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nFINAL TEST EVALUATION\n{'='*60}")
    ckpt = torch.load(
        os.path.join(SAVE_DIR, 'best_model.pth'), weights_only=False
    )
    teacher.load_state_dict(ckpt['teacher'])
    print(f"Loaded best model from epoch {ckpt['epoch'] + 1}\n")

    test_aucs = []
    for i, name in enumerate(DATASETS):
        auc = evaluate(
            teacher, test_loaders[i], i, num_classes_list[i], DEVICE
        )
        test_aucs.append(auc)
        print(f'  {name:20s}  Test AUC: {auc:.4f}')

    mean_auc = float(np.mean(test_aucs))
    print(f'\n  Mean Test AUC : {mean_auc:.4f}')
    print(f'  Best Val  AUC : {best_avg_auc:.4f}')

    with open(log_path, 'a') as f:
        f.write('\nFINAL TEST:\n')
        for name, auc in zip(DATASETS, test_aucs):
            f.write(f'  {name}: {auc:.4f}\n')
        f.write(f'Mean Test AUC: {mean_auc:.4f}\n')

    print(f'\nLog saved → {log_path}')
    print('=' * 60)
    return history, test_aucs


history, test_aucs = main()
