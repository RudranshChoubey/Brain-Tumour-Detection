# Brain Tumour Segmentation with U-Net (PyTorch)

Computer Vision course project. A small U-Net takes a brain MRI slice and predicts a binary mask: tumour vs. not tumour.

## Files

| File | Purpose |
|---|---|
| `brain_tumour_unet.py` | Loads the data, builds the U-Net, trains it, evaluates on the test set, saves the figure and the model |
| `demo.py` | Loads `segmenter.pt` and shows the MRI, the predicted mask and a red overlay for one slice |
| `segmenter.pt` | Trained weights, plus the list of test-slice paths used by the demo |
| `results.png` | Figure with 5 test slices: MRI, ground-truth mask, predicted mask |
| `training_log.txt` | Console output of the training run reported below |
| `data/` | The LGG MRI Segmentation dataset (not part of the code) |

## Setup

```
pip install torch numpy pillow matplotlib
```

The dataset is `mateuszbuda/lgg-mri-segmentation` from Kaggle, unzipped into `data/`. The code finds every `*_mask.tif` under `data/` and pairs it with its image automatically.

## Running

```
python brain_tumour_unet.py          # train, evaluate, save results.png and segmenter.pt
python demo.py                       # demo on a random test slice
python demo.py path/to/slice.tif     # demo on a slice you choose
```

Training uses the Mac GPU (`mps`) if available, then CUDA, then the CPU.

The demo works on any computer, wherever the project folder is placed:
- Train once with `python brain_tumour_unet.py`. This creates a `segmenter.pt` for your machine, and `python demo.py` then works straight away.
- The `segmenter.pt` in this repository is already trained, so you can skip training. Download the dataset into `data/` and run `python demo.py`.
- With no dataset, give the demo any MRI slice: `python demo.py path/to/slice.tif`.
- `segmenter.pt` stores the test slices as `patient_folder/slice.tif`. The demo looks for them inside `data/`, however the dataset was unzipped.

## Method

**Data.** The dataset has 3,929 slices from 110 patients (`.tif`, 256x256, 3 channels: pre-contrast, FLAIR, post-contrast), each with a mask file. Everything is resized to 128x128 (bilinear for images, nearest-neighbour for masks) and images are scaled to 0-1.

**Subset and split.**
- Patients are split first (80% train / 10% validation / 10% test, fixed random seed), so slices from one patient are never in more than one set: 77 train, 11 validation and 22 test patients.
- Slices are then sampled inside each group, mostly with tumour and some without:
  - train: 240 tumour + 80 empty = 320 slices
  - validation: 30 tumour + 10 empty = 40 slices
  - test: 60 tumour + 20 empty = 80 slices

**Model.** U-Net with three down levels and a bottleneck, using channels 32-64-128-256.
- Each block is Conv3x3 + BatchNorm + ReLU, twice.
- The encoder downsamples with max-pooling. The decoder upsamples with `ConvTranspose2d` and joins the encoder features through skip connections with `torch.cat`.
- A 1x1 convolution with a sigmoid gives one output channel: the tumour probability per pixel.

**Loss.** Binary cross-entropy + soft Dice loss. Tumours cover a tiny part of the image, so BCE alone would let the model predict "no tumour" everywhere. Dice loss penalises that directly.

**Training.** Adam, starting learning rate 1e-3 with cosine decay to zero, batch size 16, 60 epochs. The training loss and the validation Dice are printed every epoch. Three things keep the small training set from limiting the model too much:
- **Augmentation.** Every training batch gets a random flip, rotation (up to 15 degrees), zoom (0.9-1.1), shift (up to 10%) and brightness change. The same geometric change is applied to the image and its mask.
- **Learning-rate decay.** The cosine schedule removes the jitter in the loss that a constant rate caused.
- **Best-epoch selection.** The weights from the epoch with the highest validation Dice are kept, not simply the last epoch. The test set plays no part in this choice.

**Evaluation.** The probability map is thresholded at 0.5. Dice and IoU for the tumour class are computed over all test pixels pooled together. Per-slice Dice is undefined on slices with no tumour, so pooling avoids that. Pixel accuracy is deliberately not reported: predicting "no tumour" everywhere already scores about 98% on this data.

## Results

Test set: 80 slices from 22 patients that were never seen in training.

| Metric (tumour class) | Value |
|---|---|
| Dice | **0.7756** |
| IoU | **0.6334** |

The best validation Dice was 0.8647. Training loss (BCE + Dice) fell from 1.498 at epoch 1 to 0.156 at epoch 60, with the cosine schedule smoothing it out in the later epochs. The full per-epoch log is in `training_log.txt`.

On the 80 test slices, 11 of the 60 slices with a tumour got an empty prediction, and 2 of the 20 empty slices got a false tumour.

![Test predictions](results.png)

The figure shows the first 5 test slices that contain a tumour (the MRI is displayed from the FLAIR channel):
- Slices 1 and 2: the prediction closely matches the ground truth.
- Slice 3: a good match, with a small gap along the left edge.
- Slice 5: the tumour is found, but the prediction has a hole where the tumour's dark centre is. The ground truth is solid.
- Slice 4: a thin tumour that is missed. The predicted mask is empty.

**Compared with the first version of this project** (no augmentation, constant learning rate, 35 epochs, last-epoch weights), measured on the same test slices:

| Test slices (80 from the same 22 test patients) | First version, Dice | This version, Dice |
|---|---|---|
| The slices above | 0.7654 | 0.7756 |
| A second random draw of 80 slices | 0.7304 | 0.7704 |

The gain is real but modest (+0.01 and +0.04), and the two draws differ by more than the gain on the first. Training on 1,200 slices instead of 320 raised Dice on the second draw to 0.7947, but the assignment asks for about 400 slices, so that setting is not used here.

## Limitations

- **Small experiment.** The test set is 80 slices and the model was trained once with one random seed, so the Dice of 0.78 would move if the patient split or seed changed. The second test draw above scored 0.04 lower for the first version.
- **Validation set is also small.** It has 40 slices from 11 patients, and its best Dice (0.86) is higher than the test Dice (0.78). Picking the best epoch on it makes it a slightly optimistic score, which is why the test set is kept separate.
- **Pooled metric.** Pooled Dice and IoU weight large tumours more than small ones.
- **Limited training data.** The assignment's roughly 400-slice subset means 320 training slices. More slices helped in a side experiment (see Results).
- **Only 128x128 input.** Downscaling from 256x256 removes detail, which can hurt thin or small tumours.
