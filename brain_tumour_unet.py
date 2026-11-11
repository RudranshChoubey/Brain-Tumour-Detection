import os
import copy
import glob
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 128


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255


def load_mask(path):
    mask = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0)


# ==================== Model ====================
def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(3, 32)
        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)
        self.bottleneck = block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


def dice_loss(pred, target):
    inter = (pred * target).sum()
    return 1 - (2 * inter + 1) / (pred.sum() + target.sum() + 1)


def augment(x, y):
    n = x.size(0)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * 15 * math.pi / 180
    scale = 0.9 + 0.2 * torch.rand(n, device=x.device)
    flip = (torch.rand(n, device=x.device) < 0.5).float() * 2 - 1
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], dim=1),
                         torch.stack([sin * flip, cos, shift[:, 1]], dim=1)], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    y = (F.grid_sample(y, grid, align_corners=False) > 0.5).float()
    x = x * (0.9 + 0.2 * torch.rand(n, 1, 1, 1, device=x.device))
    return x, y


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 16].to(device)).cpu() for i in range(0, len(X), 16)]) > 0.5


def dice_iou(pred, target):
    pred = pred.float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (2 * inter / (pred.sum() + target.sum() + 1e-8)).item(), (inter / (union + 1e-8)).item()


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    device = get_device()
    print("Device:", device)

    # ==================== Dataset ====================
    mask_files = {}
    for m in glob.glob(os.path.join(HERE, "data", "**", "*_mask.tif"), recursive=True):
        mask_files[os.path.basename(m)] = m

    pairs = []
    for m in sorted(mask_files.values()):
        patient = os.path.basename(os.path.dirname(m))
        has_tumour = np.array(Image.open(m)).max() > 0
        pairs.append((m.replace("_mask.tif", ".tif"), m, patient, has_tumour))
    print("Found", len(pairs), "image/mask pairs")

    patients = sorted(set(p[2] for p in pairs))
    random.shuffle(patients)
    n_train = int(0.8 * len(patients))
    n_val = int(0.1 * len(patients))
    test_patients = patients[n_train:]
    val_patients = patients[n_train - n_val:n_train]
    train_patients = patients[:n_train - n_val]
    print("Patients: train", len(train_patients), "val", len(val_patients), "test", len(test_patients))


    def pick(patient_list, n_tumour, n_empty):
        tumour = [p for p in pairs if p[2] in patient_list and p[3]]
        empty = [p for p in pairs if p[2] in patient_list and not p[3]]
        return random.sample(tumour, min(n_tumour, len(tumour))) + random.sample(empty, min(n_empty, len(empty)))


    train_pairs = pick(train_patients, 240, 80)
    val_pairs = pick(val_patients, 30, 10)
    test_pairs = pick(test_patients, 60, 20)
    print("Slices: train", len(train_pairs), "val", len(val_pairs), "test", len(test_pairs))

    X_train = torch.stack([load_image(p[0]) for p in train_pairs])
    Y_train = torch.stack([load_mask(p[1]) for p in train_pairs])
    X_val = torch.stack([load_image(p[0]) for p in val_pairs])
    Y_val = torch.stack([load_mask(p[1]) for p in val_pairs])
    X_test = torch.stack([load_image(p[0]) for p in test_pairs])
    Y_test = torch.stack([load_mask(p[1]) for p in test_pairs])
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)

    # ==================== Training ====================
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_dice = -1

    for epoch in range(epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = augment(x.to(device), y.to(device))
            pred = model(x)
            loss = F.binary_cross_entropy(pred, y) + dice_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        scheduler.step()
        val_dice, _ = dice_iou(predict(model, X_val, device), Y_val)
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch + 1}/{epochs}  train loss: {total / len(X_train):.4f}  val dice: {val_dice:.4f}")

    # ==================== Evaluation ====================
    model.load_state_dict(best_state)
    preds = predict(model, X_test, device).float()
    dice, iou = dice_iou(preds, Y_test)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Test Dice (tumour): {dice:.4f}")
    print(f"Test IoU  (tumour): {iou:.4f}")

    shown = [i for i in range(len(X_test)) if Y_test[i].max() > 0][:5]
    fig, axes = plt.subplots(3, len(shown), figsize=(3 * len(shown), 9), squeeze=False)
    for col, i in enumerate(shown):
        axes[0, col].imshow(X_test[i, 1], cmap="gray")
        axes[1, col].imshow(Y_test[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("MRI image")
    axes[1, 0].set_ylabel("Ground truth")
    axes[2, 0].set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "results.png"), dpi=150)

    torch.save({"model": model.cpu().state_dict(), "test_files": [p[2] + "/" + os.path.basename(p[0]) for p in test_pairs]},
               os.path.join(HERE, "segmenter.pt"))
    print("Saved results.png and segmenter.pt")
