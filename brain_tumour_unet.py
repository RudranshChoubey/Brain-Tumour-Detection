import os
import glob
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
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]
    print("Patients: train", len(train_patients), "test", len(test_patients))


    def pick(patient_list, n_tumour, n_empty):
        tumour = [p for p in pairs if p[2] in patient_list and p[3]]
        empty = [p for p in pairs if p[2] in patient_list and not p[3]]
        return random.sample(tumour, min(n_tumour, len(tumour))) + random.sample(empty, min(n_empty, len(empty)))


    train_pairs = pick(train_patients, 240, 80)
    test_pairs = pick(test_patients, 60, 20)
    print("Slices: train", len(train_pairs), "test", len(test_pairs))

    X_train = torch.stack([load_image(p[0]) for p in train_pairs])
    Y_train = torch.stack([load_mask(p[1]) for p in train_pairs])
    X_test = torch.stack([load_image(p[0]) for p in test_pairs])
    Y_test = torch.stack([load_mask(p[1]) for p in test_pairs])
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)

    # ==================== Training ====================
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 35

    for epoch in range(epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.binary_cross_entropy(pred, y) + dice_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch + 1}/{epochs}  train loss: {total / len(X_train):.4f}")

    # ==================== Evaluation ====================
    model.eval()
    with torch.no_grad():
        probs = torch.cat([model(X_test[i:i + 16].to(device)).cpu() for i in range(0, len(X_test), 16)])
    preds = (probs > 0.5).float()

    inter = (preds * Y_test).sum()
    union = preds.sum() + Y_test.sum() - inter
    dice = 2 * inter / (preds.sum() + Y_test.sum() + 1e-8)
    iou = inter / (union + 1e-8)
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

    torch.save({"model": model.cpu().state_dict(), "test_files": [p[0] for p in test_pairs]},
               os.path.join(HERE, "segmenter.pt"))
    print("Saved results.png and segmenter.pt")
