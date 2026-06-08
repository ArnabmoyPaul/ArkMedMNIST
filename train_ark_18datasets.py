import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import medmnist
from medmnist import INFO
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import os
import time

# ============================================================
# CONFIG  — edit these before running
# ============================================================
EXPERIMENT   = "ark_all_18_datasets"
EPOCHS       = 25
PATIENCE     = 5
LR           = 1e-3
MOMENTUM_EMA = 0.9
IMAGE_SIZE   = 64
BATCH_SIZE   = 8
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR     = "./outputs/" + EXPERIMENT
CHECKPOINT   = os.path.join(SAVE_DIR, "checkpoint_latest.pth")   # resume from here
BEST_MODEL   = os.path.join(SAVE_DIR, "best_model.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 18 datasets: 12 × 2D + 6 × 3D ──────────────────────────────────────────
DATASETS_2D = [
    'PathMNIST', 'BloodMNIST', 'DermaMNIST', 'OCTMNIST',
    'PneumoniaMNIST', 'RetinaMNIST', 'BreastMNIST', 'TissueMNIST',
    'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST', 'ChestMNIST',
]
DATASETS_3D = [
    'OrganMNIST3D', 'NoduleMNIST3D', 'AdrenalMNIST3D',
    'FractureMNIST3D', 'VesselMNIST3D', 'SynapseMNIST3D',
]
DATASETS = DATASETS_2D + DATASETS_3D

# ── Class counts ─────────────────────────────────────────────────────────────
NUM_CLASSES = {
    # 2D
    'PathMNIST': 9, 'BloodMNIST': 8, 'DermaMNIST': 7, 'OCTMNIST': 4,
    'PneumoniaMNIST': 2, 'RetinaMNIST': 5, 'BreastMNIST': 3,
    'TissueMNIST': 8, 'OrganAMNIST': 11, 'OrganCMNIST': 11,
    'OrganSMNIST': 11, 'ChestMNIST': 14,
    # 3D
    'OrganMNIST3D': 11, 'NoduleMNIST3D': 2, 'AdrenalMNIST3D': 2,
    'FractureMNIST3D': 3, 'VesselMNIST3D': 2, 'SynapseMNIST3D': 2,
}

TASK_TYPE = {
    # 2D — multi-class except ChestMNIST (multi-label)
    'PathMNIST': 'multi-class', 'BloodMNIST': 'multi-class',
    'DermaMNIST': 'multi-class', 'OCTMNIST': 'multi-class',
    'PneumoniaMNIST': 'multi-class', 'RetinaMNIST': 'multi-class',
    'BreastMNIST': 'multi-class', 'TissueMNIST': 'multi-class',
    'OrganAMNIST': 'multi-class', 'OrganCMNIST': 'multi-class',
    'OrganSMNIST': 'multi-class', 'ChestMNIST': 'multi-label',
    # 3D — all multi-class
    'OrganMNIST3D': 'multi-class', 'NoduleMNIST3D': 'multi-class',
    'AdrenalMNIST3D': 'multi-class', 'FractureMNIST3D': 'multi-class',
    'VesselMNIST3D': 'multi-class', 'SynapseMNIST3D': 'multi-class',
}
# ============================================================


# ── 2D Dataset wrapper ───────────────────────────────────────────────────────
MEDMNIST_2D_MAP = {
    'PathMNIST': medmnist.PathMNIST, 'BloodMNIST': medmnist.BloodMNIST,
    'DermaMNIST': medmnist.DermaMNIST, 'OCTMNIST': medmnist.OCTMNIST,
    'PneumoniaMNIST': medmnist.PneumoniaMNIST, 'RetinaMNIST': medmnist.RetinaMNIST,
    'BreastMNIST': medmnist.BreastMNIST, 'TissueMNIST': medmnist.TissueMNIST,
    'OrganAMNIST': medmnist.OrganAMNIST, 'OrganCMNIST': medmnist.OrganCMNIST,
    'OrganSMNIST': medmnist.OrganSMNIST, 'ChestMNIST': medmnist.ChestMNIST,
}

def _transform(size, train):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class Dataset2D(Dataset):
    def __init__(self, name, split, size=64):
        self.name = name
        self.task = TASK_TYPE[name]
        self.nc   = NUM_CLASSES[name]
        self.data = MEDMNIST_2D_MAP[name](split=split, download=True,
                                          size=size, as_rgb=True)
        self.tf_train = _transform(size, True)
        self.tf_val   = _transform(size, False)
        self.train    = (split == 'train')

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        lbl = torch.tensor(lbl).squeeze()
        if self.task == 'multi-class':
            lbl = lbl.long()
        else:
            lbl = lbl.float()
        tf  = self.tf_train if self.train else self.tf_val
        return tf(img), tf(img), lbl


# ── 3D Dataset wrapper ───────────────────────────────────────────────────────
MEDMNIST_3D_MAP = {
    'OrganMNIST3D': medmnist.OrganMNIST3D, 'NoduleMNIST3D': medmnist.NoduleMNIST3D,
    'AdrenalMNIST3D': medmnist.AdrenalMNIST3D, 'FractureMNIST3D': medmnist.FractureMNIST3D,
    'VesselMNIST3D': medmnist.VesselMNIST3D, 'SynapseMNIST3D': medmnist.SynapseMNIST3D,
}

class Dataset3D(Dataset):
    def __init__(self, name, split):
        self.name  = name
        self.task  = TASK_TYPE[name]
        self.nc    = NUM_CLASSES[name]
        raw        = MEDMNIST_3D_MAP[name](split=split, download=True)
        self.imgs  = raw.imgs.astype(np.float32) / 255.0
        self.lbls  = raw.labels.squeeze().astype(np.int64)
        self.train = (split == 'train')

    def __len__(self): return len(self.imgs)

    def _aug(self, vol):
        if self.train:
            for ax in range(3):
                if np.random.rand() > 0.5:
                    vol = np.flip(vol, ax).copy()
            vol = np.clip(vol * np.random.uniform(0.85,1.15)
                          + np.random.uniform(-0.05,0.05), 0, 1)
        return torch.tensor(vol[None], dtype=torch.float32)   # (1,D,H,W)

    def __getitem__(self, idx):
        v = self.imgs[idx]
        return self._aug(v), self._aug(v), torch.tensor(int(self.lbls[idx]))


# ── Model ────────────────────────────────────────────────────────────────────
class ArkMedMNIST18(nn.Module):
    """
    Swin-Base backbone shared across all 18 datasets.
    3D volumes are folded: (B,1,D,H,W) → (B*D,3,H,W) before encoding.
    Each dataset has its own classification head.
    """
    def __init__(self, num_classes_list, img_size=64):
        super().__init__()
        self.encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            img_size=img_size,
            num_classes=0,
            global_pool="avg",
        )
        self.num_features = self.encoder.num_features
        self.omni_heads   = nn.ModuleList([
            nn.Linear(self.num_features, nc)
            for nc in num_classes_list
        ])

    def _encode(self, x):
        if x.dim() == 5:                    # 3D: (B,1,D,H,W)
            B, C, D, H, W = x.shape
            # fold depth into batch → (B*D, 1, H, W)
            x = x.permute(0,2,1,3,4).reshape(B*D, C, H, W)
            # replicate to 3 channels
            x = x.expand(-1, 3, -1, -1)
            # upsample 28×28 → 64×64 to match model input size
            if H != IMAGE_SIZE:
                x = torch.nn.functional.interpolate(
                    x, size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode='bilinear', align_corners=False)
            feats = self.encoder(x)          # (B*D, F)
            return feats.view(B, D, -1).mean(1)   # (B, F)
        if x.shape[1] == 1:                 # 2D grayscale
            x = x.expand(-1, 3, -1, -1)
        return self.encoder(x)

    def forward(self, x, head_n):
        feats = self._encode(x)
        return feats, self.omni_heads[head_n](feats)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ema_update(student, teacher, momentum):
    with torch.no_grad():
        for s, t in zip(student.parameters(), teacher.parameters()):
            t.data = momentum * t.data + (1 - momentum) * s.data


def compute_auc(y_true, y_pred, nc, task):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    aucs   = []
    if task == 'multi-class':
        oh = np.zeros((len(y_true), nc))
        for i, v in enumerate(y_true):
            oh[i, int(v)] = 1
        for c in range(nc):
            if oh[:, c].sum() > 0:
                aucs.append(roc_auc_score(oh[:, c], y_pred[:, c]))
    else:
        for c in range(nc):
            if y_true[:, c].sum() > 0:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def train_one_cycle(model, teacher, name, loader, head_n, optimizer, epoch, device):
    model.train()
    crit = (nn.CrossEntropyLoss() if TASK_TYPE[name] == 'multi-class'
            else nn.BCEWithLogitsLoss())
    MSE  = nn.MSELoss()
    coff = min((epoch / EPOCHS) * 0.1, 0.1)
    total_loss = total_cls = total_mse = n = 0
    for v1, v2, lbl in loader:
        v1, v2, lbl = v1.to(device), v2.to(device), lbl.to(device)
        feat_s, pred_s = model(v1, head_n)
        with torch.no_grad():
            feat_t, _ = teacher(v2, head_n)
        loss_cls  = crit(pred_s, lbl)
        loss_mse  = MSE(feat_s, feat_t)
        loss      = (1 - coff) * loss_cls + coff * loss_mse
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); total_cls += loss_cls.item()
        total_mse  += loss_mse.item(); n += 1
    return total_loss/n, total_cls/n, total_mse/n


def evaluate_auc(model, name, loader, head_n, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for v1, _, lbl in loader:
            v1 = v1.to(device)
            _, out = model(v1, head_n)
            if TASK_TYPE[name] == 'multi-class':
                out = torch.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)
            preds.append(out.cpu()); labels.append(lbl.cpu())
    return compute_auc(torch.cat(labels), torch.cat(preds),
                       NUM_CLASSES[name], TASK_TYPE[name])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXPERIMENT :", EXPERIMENT)
    print("DATASETS   : 12 × 2D  +  6 × 3D  =  18 total")
    print("EPOCHS     :", EPOCHS, "  PATIENCE:", PATIENCE)
    print("DEVICE     :", DEVICE)
    print("=" * 60)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    print("\nBuilding dataloaders...")
    train_loaders, val_loaders, test_loaders = [], [], []

    for name in DATASETS:
        is3d = name in DATASETS_3D
        Cls  = Dataset3D if is3d else Dataset2D

        train_ds = Cls(name, 'train') if is3d else Cls(name, 'train', IMAGE_SIZE)
        val_ds   = Cls(name, 'val')   if is3d else Cls(name, 'val',   IMAGE_SIZE)
        test_ds  = Cls(name, 'test')  if is3d else Cls(name, 'test',  IMAGE_SIZE)

        train_loaders.append(DataLoader(train_ds, batch_size=BATCH_SIZE,
            shuffle=True,  num_workers=0, pin_memory=True))
        val_loaders.append(DataLoader(val_ds,   batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=True))
        test_loaders.append(DataLoader(test_ds,  batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=True))
        print(f"  {name}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    num_classes_list = [NUM_CLASSES[d] for d in DATASETS]
    model   = ArkMedMNIST18(num_classes_list, IMAGE_SIZE).to(DEVICE)
    teacher = ArkMedMNIST18(num_classes_list, IMAGE_SIZE).to(DEVICE)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.load_state_dict(model.state_dict())
    model.encoder.set_grad_checkpointing(True)

    print(f"  Feature dim : {model.num_features}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1_000_000}M")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_avg_auc  = 0.0
    patience_ctr  = 0
    stopped_epoch = EPOCHS

    if os.path.isfile(CHECKPOINT):
        print(f"\n>>> Resuming from checkpoint: {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_avg_auc  = ckpt['best_avg_auc']
        patience_ctr  = ckpt['patience_ctr']
        print(f"    Resumed from epoch {start_epoch-1+1}  |  best_auc={best_avg_auc:.4f}  |  patience={patience_ctr}")
    else:
        print("\n>>> No checkpoint found — starting fresh")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = os.path.join(SAVE_DIR, "train_log.txt")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write(f"Experiment: {EXPERIMENT}\n")
            f.write(f"Datasets: {DATASETS}\n")
            f.write(f"Epochs: {EPOCHS}  Patience: {PATIENCE}\n\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting cyclic pretraining (18 datasets)...")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}   [patience {patience_ctr}/{PATIENCE}]")
        print(f"{'='*60}")

        # Train — cycle through all 18 datasets
        for i, name in enumerate(DATASETS):
            loss, cls_l, mse_l = train_one_cycle(
                model, teacher, name,
                train_loaders[i], i, optimizer, epoch, DEVICE)
            ema_update(model, teacher, MOMENTUM_EMA)
            tag = "3D" if name in DATASETS_3D else "2D"
            print(f"  [{tag}] {name}: loss={round(loss,4)} cls={round(cls_l,4)} mse={round(mse_l,4)}")

        scheduler.step()

        # Validate
        print("\n  Validation AUC:")
        auc_list = []
        for i, name in enumerate(DATASETS):
            auc = evaluate_auc(teacher, name, val_loaders[i], i, DEVICE)
            auc_list.append(auc)
            tag = "3D" if name in DATASETS_3D else "2D"
            print(f"    [{tag}] {name}: {round(auc, 4)}")

        avg_auc    = float(np.mean(auc_list))
        auc_2d     = float(np.mean([auc_list[i] for i,n in enumerate(DATASETS) if n in DATASETS_2D]))
        auc_3d     = float(np.mean([auc_list[i] for i,n in enumerate(DATASETS) if n in DATASETS_3D]))
        epoch_time = time.time() - t0

        print(f"    --- 2D avg: {round(auc_2d,4)}  |  3D avg: {round(auc_3d,4)}  |  Overall: {round(avg_auc,4)} ---")
        print(f"  Epoch time: {round(epoch_time/60, 1)} min")

        # Log
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1}: avg={round(avg_auc,4)} 2D={round(auc_2d,4)} 3D={round(auc_3d,4)} time={round(epoch_time/60,1)}min\n")
            for name, auc in zip(DATASETS, auc_list):
                f.write(f"  {name}: {round(auc,4)}\n")
            f.write("\n")

        # Save checkpoint every epoch (for resume after power cut)
        torch.save({
            'epoch':        epoch,
            'state_dict':   model.state_dict(),
            'teacher':      teacher.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'best_avg_auc': best_avg_auc,
            'patience_ctr': patience_ctr,
            'auc_list':     auc_list,
            'datasets':     DATASETS,
        }, CHECKPOINT)

        # Save best
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            patience_ctr = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'teacher':    teacher.state_dict(),
                'avg_auc':    avg_auc,
                'auc_2d':     auc_2d,
                'auc_3d':     auc_3d,
                'auc_list':   auc_list,
                'datasets':   DATASETS,
            }, BEST_MODEL)
            print(f"  ✓ New best saved: overall={round(avg_auc,4)}")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience: {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                stopped_epoch = epoch + 1
                print(f"\n  EARLY STOPPING at epoch {stopped_epoch}")
                with open(log_path, "a") as f:
                    f.write(f"Early stopping at epoch {stopped_epoch}\n")
                break

    # ── Final test ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")

    ckpt = torch.load(BEST_MODEL, weights_only=False)
    teacher.load_state_dict(ckpt['teacher'])
    print(f"Best model from epoch {ckpt['epoch']+1}")

    test_aucs = []
    for i, name in enumerate(DATASETS):
        auc = evaluate_auc(teacher, name, test_loaders[i], i, DEVICE)
        test_aucs.append(auc)
        tag = "3D" if name in DATASETS_3D else "2D"
        print(f"  [{tag}] {name}: {round(auc,4)}")

    mean_2d = np.mean([test_aucs[i] for i,n in enumerate(DATASETS) if n in DATASETS_2D])
    mean_3d = np.mean([test_aucs[i] for i,n in enumerate(DATASETS) if n in DATASETS_3D])
    mean_all = np.mean(test_aucs)

    print(f"\n  2D mean AUC  : {round(float(mean_2d),4)}")
    print(f"  3D mean AUC  : {round(float(mean_3d),4)}")
    print(f"  Overall mean : {round(float(mean_all),4)}")
    print(f"  Best val AUC : {round(best_avg_auc,4)}")
    print(f"  Stopped epoch: {stopped_epoch}")

    with open(log_path, "a") as f:
        f.write("\nFINAL TEST RESULTS:\n")
        for name, auc in zip(DATASETS, test_aucs):
            f.write(f"  {name}: {round(auc,4)}\n")
        f.write(f"2D mean: {round(float(mean_2d),4)}\n")
        f.write(f"3D mean: {round(float(mean_3d),4)}\n")
        f.write(f"Overall: {round(float(mean_all),4)}\n")

    print(f"\nLog saved to {log_path}")
    print("=" * 60)


main()
