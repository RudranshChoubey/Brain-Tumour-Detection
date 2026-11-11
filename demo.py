import os
import sys
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from brain_tumour_unet import UNet, load_image, get_device, HERE

device = get_device()
checkpoint = torch.load(os.path.join(HERE, "segmenter.pt"), map_location="cpu")
model = UNet()
model.load_state_dict(checkpoint["model"])
model.to(device).eval()

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    matches = glob.glob(os.path.join(HERE, "data", "**", random.choice(checkpoint["test_files"])), recursive=True)
    if not matches:
        sys.exit("Test slice not found. Put the dataset in the data folder or pass an image path.")
    path = matches[0]
print("Slice:", path)

x = load_image(path)
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device))[0, 0].cpu().numpy() > 0.5

mri = x[1].numpy()
overlay = np.stack([mri, mri, mri], axis=-1)
overlay[pred] = [1, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(mri, cmap="gray")
axes[0].set_title("MRI")
axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Predicted mask")
axes[2].imshow(overlay)
axes[2].set_title("Overlay")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
