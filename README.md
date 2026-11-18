# DIT5411 — Handwritten Traditional Chinese Character Recognition

**Course:** THEi DIT5411 Machine Learning — Assignment (Character Recognition)
**Author:** `<Your Name>` (`<Student ID>`)
**Framework:** TensorFlow 2.21 / Keras 3.13 · **Environment:** Google Colab, Tesla T4 (16 GB)
**Notebook:** [`DIT5411_Character_Recognition.ipynb`](DIT5411_Character_Recognition.ipynb)

---

## 1. Overview

This project trains a convolutional neural network that recognises handwritten
**Traditional Chinese characters** across **13,065 character classes** — a large-scale,
fine-grained classification problem rather than a toy digit task.

The dataset provides only ~52 samples per character, which is far too little to train a
13,065-way classifier directly. The assignment therefore asks for three things, all
implemented here:

1. split each class into **first 40 samples for training / the remainder for testing**,
2. **augment** the training images (rotation, shearing, scaling, translation) so that
   every class has **at least 200 training samples**,
3. define **several CNN architectures** and report the recognition rate of each on the
   test set.

Three models were trained and evaluated on **162,550 test images**. The best model
(`vgg_mini`) reaches **94.66 % Top-1 / 99.02 % Top-5**, while the smallest model
(`resnet_mini`, 1.99 M parameters) reaches **92.30 % / 98.37 %** with 3.5× fewer
parameters than the best one.

---

## 2. Dataset

| Item | Value |
|---|---|
| Source | [AI-FREE-Team / Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) |
| Image archive | [chenkenanalytic / handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all) (12 split parts, ≈1.1 GB) |
| Classes | 13,065 |
| Images | 684,677 (average 52 per class) |
| Image size | 300 × 300, PNG, white background / black ink |
| Licence | **CC BY-NC-SA 4.0** |

```bash
git clone --depth 1 https://github.com/chenkenanalytic/handwritting_data_all.git
cat handwritting_data_all/all_data.zip.0* > all_data.zip
unzip -q -O big5 all_data.zip -d /content        # folder names are Big5-encoded
```

> `-O big5` is mandatory: the character folders are stored in Big5, and a plain `unzip`
> produces mojibake names. In the extraction used for this run the folders are instead
> named with numeric class IDs (13,065 folders, one per character); the notebook treats
> the folder name as the class label either way.

### Split actually used

| Set | Images | Rule |
|---|---|---|
| Train (original) | 520,880 | first 40 images of each class |
| Train (after augmentation) | **2,604,400** | 200 images per class |
| Test | **162,550** | all remaining images of each class |

13,065 folders were found but **13,022 classes** were used: 43 classes contain fewer than
45 images and cannot provide both a 40-image training set and a test set, so they were
dropped. `13,022 × 40 = 520,880` original training images; `13,022 × 200 = 2,604,400`
after augmentation.

---

## 3. Image pre-processing

1. read as **grayscale**;
2. normalise orientation to **white background / black ink** (the mean intensity is
   checked and the image is inverted if needed, so mixed polarity in the source data
   cannot confuse the model);
3. resize to **48 × 48** with area interpolation;
4. scale pixels to `[0, 1]` inside the input pipeline (`uint8` kept in storage).

The whole training set is stored once as a compact `uint8` **memmap** (~6 GB) instead of
being re-decoded from 2.6 M PNG files every epoch. A Python generator yields one batch at
a time to `tf.data`, so the 12.7 GB Colab RAM is never exceeded and the CPU prefetch
overlaps with GPU work.

---

## 4. Data augmentation

Each original training image produces **4 augmented copies**, therefore
`40 × (1 + 4) = 200` samples per class.

| Transformation | Range | Why |
|---|---|---|
| Rotation | ±12° | handwriting is often slightly tilted |
| Shearing | ±0.12 | leaning / italic strokes |
| Scaling | 0.88 – 1.12 | different pen sizes and writing sizes |
| Translation | ±6 % of the image size | the character is rarely perfectly centred |
| Gamma (contrast) | 0.80 – 1.25 | pen thickness, scan/ink quality |
| Border fill | constant white | prevents black corners after warping |

Implemented with OpenCV affine warps (`cv2.getRotationMatrix2D` + `cv2.warpAffine`),
following the OpenCV reference listed in §12. One random combination is drawn per copy, so
the four copies of an image differ from each other.

**Horizontal and vertical flipping are deliberately not used.** A mirrored Chinese
character is a different character (or does not exist at all), so a flip would silently
corrupt the label and cap the achievable accuracy. This is the most common mistake for
this assignment, and it is the main reason augmentation must be geometric-warp based here
rather than the mirror-based pipeline used for MNIST.

![augmentation examples](augmentation_examples.png)

*One original plus its four augmented copies, for three different classes.*

The one-off cost of building this data set is **9.4 minutes** (684,677 images read,
520,880 warped 4× and written to the memmap), measured on the Colab runtime.

---

## 5. Model architectures

All three models receive a `48 × 48 × 1` image and end in a **13,022-neuron softmax**
output layer — one neuron per class, as required by the assignment.

| # | Model | Structure | Layers | Parameters |
|---|---|---|---|---|
| A | `cnn_baseline` | Conv32 → pool → Conv64 → pool → Conv128 → pool → Flatten → Dense256 → Dropout 0.3 → Softmax | 11 | 4,619,230 |
| B | `vgg_mini` | 3 × [Conv-BN-ReLU ×2 → pool] (32/64/128) → GlobalAveragePooling → Dense512 → Dropout 0.5 → Softmax | 26 | 7,034,110 |
| C | `resnet_mini` | stem Conv32-BN + 3 residual blocks (32/1, 64/2, 128/2), each 2×Conv-BN + identity shortcut (1×1 conv when shapes change) → GAP → Dropout 0.4 → Softmax | 32 | 1,987,966 |

Design intent:

- **A** is the simplest thing that could work: a plain conv stack with a `Flatten` +
  dense head. It is the baseline that shows the cost of not using normalisation.
- **B** adds BatchNorm in every block and replaces `Flatten` with global average pooling,
  which deletes the 1.2 M-parameter dense layer in front of the classifier; almost all of
  B's 7.0 M parameters sit in the wide 512 → 13,022 classifier, not in the convolutions
  (only ~0.37 M).
- **C** tests whether residual connections help when the parameter budget is small; it is
  the deepest network (32 layers) with the fewest parameters.

---

## 6. Training setup

| Setting | Value |
|---|---|
| Optimiser | Adam — lr 3e-4 (A, C), 1e-3 (B); `ReduceLROnPlateau(factor 0.3, patience 1, min_lr 1e-5)` |
| Loss | sparse categorical cross-entropy, label smoothing 0.1 |
| Batch size | 512 |
| Steps per epoch | 5,086 (`2,604,400 / 512`) |
| Epochs | 10 for all three models (EarlyStopping patience 2 never triggered) |
| Mixed precision | `mixed_float16` (T4), output layer kept in `float32` for stable softmax |
| Seed | 42 |
| Monitoring | accuracy on a fixed 20,000-image validation subset of the test set |
| Checkpoints | best weights per epoch saved; the best epoch is restored at the end |

Hyperparameters are disclosed per model because the learning rate had to be chosen per
architecture to keep training stable (see §8). A and C share the same base learning rate,
so their comparison is directly comparable; B succeeded with a **larger** learning rate.

---

## 7. Results

Full test set: **162,550 images never used for gradient updates** (except that the 20 k
validation subset is drawn from it — see §9).

| Model | Test Top-1 | Test Top-5 | Parameters | Training time (10 epochs) | Per step |
|---|---|---|---|---|---|
| A `cnn_baseline` | **80.08 %** | 95.16 % | 4,619,230 | 16.9 min | 20 ms |
| B `vgg_mini` | **94.66 %** | **99.02 %** | 7,034,110 | 37.5 min | 44 ms |
| C `resnet_mini` | **92.30 %** | 98.37 % | 1,987,966 | 49.7 min | 59 ms |

Validation accuracy per epoch (20 k subset):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 14.0 | 42.1 | 56.7 | 64.3 | 69.2 | 72.5 | 75.1 | 76.8 | 77.9 | 80.1 |
| B | 0.3 | 0.1 | 2.6 | 6.2 | 12.7 | 28.4 | 0.6 | 92.2 | 54.6 | 94.7 |
| C | 1.0 | 0.1 | 1.8 | 6.8 | 83.9 | 17.8 | 90.4 | 40.6 | 91.7 | 92.4 |

Both B and C alternate between collapse and recovery. In every case the collapse occurs in
the epoch *before* a `ReduceLROnPlateau` reduction, and the epoch *after* the reduction sets
a new best — while training accuracy keeps rising monotonically (B: 13.7 % → 86.9 %,
C: 3.2 % → 78.0 %). These are optimiser / BatchNorm-calibration events, not true divergence.

![training curves](training_curves.png)
![predictions](predictions.png)

---

## 8. Discussion

**A → B: normalisation beats raw depth.** Adding BatchNorm and swapping the dense head for
global average pooling raised Top-1 from 80.08 % to 94.66 % (+14.6 points) while adding
2.4 M parameters. Model A had **not converged** when the 10-epoch budget ran out: its
validation curve climbs monotonically throughout (76.8 % → 77.9 % → 80.1 % over the last
three epochs, i.e. +3.3 points with no sign of flattening). Without normalisation its
progress per epoch is simply much slower, so the 10-epoch budget is fair to B and C but
generous to A; given 20+ epochs A would certainly score higher.

**B vs C: accuracy per parameter.** C reaches 92.30 % with **1.99 M** parameters, only 2.4
points behind B's 7.03 M. Per parameter C is ~3.5× more efficient, which is what the
residual/GAP design is supposed to buy. C is however ~35 % slower per training step
(59 ms vs 44 ms) because it has 32 layers of small kernels, so on a fixed time budget B
wins on accuracy while C wins on model size — a realistic engineering trade-off to report.

**Training stability, and what it costs.** With 13,022 output neurons the softmax layer
dominates the loss and training is sensitive to the learning rate. Two symptoms appeared:

- B, at lr 1e-3, produced `val_loss ≈ 41` after epoch 1 while its training loss was only
  5.3, then `val_loss ≈ 90` at epoch 2 — BatchNorm running statistics not yet matching
  fast-moving weights: training accuracy was already 57 % while validation accuracy was
  0.12 %. It collapsed a second time at epoch 7 (28.4 % → 0.6 %) while training accuracy
  kept climbing to 83.9 %. Each collapse was followed by a learning-rate cut
  (1e-3 → 3e-4 → 9e-5 → 2.7e-5) and each cut produced a large jump to a new best
  (92.2 %, then 94.7 %).
- C showed the violent oscillation `83.9 % → 17.8 % → 90.4 % → 40.6 % → 91.7 %`. Each
  collapse coincides with a `ReduceLROnPlateau` trigger, after which the reduced learning
  rate restored (and improved) the previous best. Without that callback C would plausibly
  have ended far below 92 %.

**Top-1 vs Top-5.** The gap between Top-1 and Top-5 is 4.4 points for B, 6.1 for C and
15.1 for A — the expected signature of a 13k-class problem: the correct class is almost
always among the five best scores, and the residual errors are characters that share
components or radicals. Top-5 is therefore the more informative measure of *learned feature
quality*, while Top-1 is what matters for a deployed recogniser. The size of A's gap (15.1)
reflects that its softmax ranking is fuzzy rather than confident.

**Is augmentation what produces the accuracy?** All three models were trained on the same
200-image-per-class set, so the table above compares architectures, not augmentation
strategies — no ablation was run. The argument for augmentation here is structural: 40
samples per class against 13,022 classes is far too little to condition a 13k-way softmax,
and the four warped copies per image regularise exactly the dimensions that vary between
writers (slant, scale, position, stroke weight) without inventing new glyphs. A controlled
ablation (same model, 40 raw images vs 200 augmented) is the natural next experiment and is
listed in §11.

---

## 9. Limitations

- **48 × 48 input.** The sources are 300 × 300; down-sampling to 48 × 48 discards stroke
  detail, and it is the main ceiling on accuracy for characters that differ only by fine
  strokes. 64 × 64 is the obvious next step (≈1.8× the data volume and compute).
- **Numeric class labels.** In the extraction used here the class folders are named with
  IDs rather than characters, so the prediction figure shows class IDs. The mapping back
  to glyphs is available from the original dataset.
- **Validation subset inside the test set.** Model selection and `ReduceLROnPlateau`
  watched a 20,000-image subset of the 162,550 test images, so a small optimistic bias is
  possible: 20,000 of the 162,550 test images (12 %) took part in model selection, while
  the remaining 142,550 were only ever used for the final measurement. A clean protocol
  would hold out a disjoint validation split.
- **Single run per model** (seed 42) — no error bars, so differences of <1 point should
  not be over-interpreted.
- **Class imbalance is mild but real**: most classes contribute 45–60 samples, and a class
  with exactly 45 samples contributes only 5 test images, so per-class accuracy on the
  smallest classes is much noisier than the aggregate number suggests.

---

## 10. How to reproduce

1. Open the notebook in Google Colab and set
   `Runtime → Change runtime type → T4 GPU`.
2. `Runtime → Run all`. Total compute in this run: **≈1 h 55 min** on one T4
   (9.4 min augmentation + 16.9 + 37.5 + 49.7 min training + ~1 min evaluation).
3. Artefacts are written to `/content/DIT5411/`: `A_cnn.weights.h5`, `B_vgg.weights.h5`,
   `C_resnet.weights.h5`, `*_history.csv`, `*_log.json`, `results.json`, and the three
   figures.

Smoke test without the full data set: set `CFG["MAX_CLASSES"] = 300` and
`CFG["EPOCHS"] = 1` (~8 min) to verify the whole pipeline end to end.

**Repository layout**

```
DIT5411-<Your Name>/
├── DIT5411_Character_Recognition.ipynb   # full pipeline, outputs included
├── README.md
├── requirements.txt
├── results.json                          # accuracy of every model
├── A_cnn_history.csv  B_vgg_history.csv  C_resnet_history.csv
├── augmentation_examples.png
├── training_curves.png
├── predictions.png
└── .gitignore                            # excludes the 1.1 GB dataset and weights
```

`requirements.txt`:

```
tensorflow>=2.15
numpy
opencv-python-headless
matplotlib
pillow
```

The dataset is **not** committed (1.1 GB); §2 shows how to fetch it.

---

## 11. Conclusion & next steps

Three architectures were trained under an identical data pipeline and epoch budget on
13,022 classes. The best model (`vgg_mini`, BatchNorm + global average pooling) reaches
**94.66 % Top-1 and 99.02 % Top-5** on 162,550 unseen images, and the smallest model
(`resnet_mini`, 1.99 M parameters) reaches 92.30 % / 98.37 %, which is the better
accuracy-per-parameter result of the three.

The practical lessons from this assignment:

1. **Augmentation is the enabling step, not a refinement.** With ~52 samples per character
   the classifier cannot be trained at all without synthesising more views of each glyph,
   and the augmentation must respect the structure of the script (no mirroring).
2. **At 13k classes, normalisation and the output head matter more than raw depth.** A
   4.6 M-parameter model without BatchNorm is 14.6 points behind a 7.0 M-parameter model
   with it, and a 2.0 M-parameter residual model beats the 4.6 M baseline by 12 points.
3. **The learning rate is the main source of instability**, visible as validation collapses
   between otherwise improving epochs; `ReduceLROnPlateau` is what made the run converge in
   10 epochs per model.

Concrete next steps, in order of expected gain:

- **Ablation on augmentation**: train one model on the raw 40 images per class to quantify
  the contribution directly (this is the one experiment this report does not include).
- **Larger input**: 64 x 64 instead of 48 x 48, which is the main accuracy ceiling for
  characters that differ by a few strokes.
- **Longer schedule / cosine decay** and a disjoint validation split instead of a subset of
  the test set.
- **Per-class error analysis** (worst-100 classes) to show which characters the model
  confuses, and test-time augmentation at inference.
- **Deployment**: quantise the best model to int8 (TFLite) and report the accuracy/latency
  trade-off — the natural bridge to the embedded-systems side of the course.

---

## 12. References

1. AI . FREE Team, *Traditional Chinese Handwriting Dataset*, GitHub, 2020 —
   <https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset>
2. AI . FREE Team, *Handwriting Chinese Characters Recognition* — reference CNN
   implementation —
   <https://github.com/AI-FREE-Team/Handwriting-Chinese-Characters-Recognition>
3. kimanalytics, *Handwritten Digit Recognition using Keras and TensorFlow* —
   <https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow>
4. Darshita Jain, *Introduction to OpenCV* (affine transformations) —
   <https://github.com/darshitajain/Introduction-to-OpenCV>
5. He, K., Zhang, X., Ren, S., Sun, J., *Deep Residual Learning for Image Recognition*,
   CVPR 2016.
6. Ioffe, S., Szegedy, C., *Batch Normalization*, ICML 2015.
7. Lin, M., Chen, Q., Yan, S., *Network In Network* (global average pooling), ICLR 2014.
8. Original data source: STUST EECS Chinese MNIST —
   <https://scidm.nchc.org.tw/dataset/stusteecs_chinese_mnist>

---

## 13. Licence & acknowledgement

The dataset is released under **CC BY-NC-SA 4.0**. It was developed by AI . FREE Team from
the *STUST EECS Chinese MNIST* dataset provided by the Department of Electronic
Engineering, Southern Taiwan University of Science and Technology. This repository uses
the dataset for academic coursework only and does not redistribute it.
