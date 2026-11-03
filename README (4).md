# 🛣️ Road Damage Detection with YOLO

Detect and localize road damage (potholes, cracks and more) in road images using YOLO object detection, then serve the best model in a small web app.

> 🔗 **Live demo:** _add your Hugging Face Space link here_

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Data Cleaning](#-data-cleaning)
4. [Models and Training Setup](#-models-and-training-setup)
5. [Results](#-results)
6. [Hyperparameter Tuning](#-hyperparameter-tuning)
7. [Experiment: Merging Confusing Classes](#-experiment-merging-confusing-classes)
8. [Challenges and Lessons Learned](#-challenges-and-lessons-learned)
9. [Deployment App](#-deployment-app)
10. [Step-by-Step Guide](#-step-by-step-guide)
11. [Limitations and Future Work](#-limitations-and-future-work)
12. [Repository Structure](#-repository-structure)

---

## 🎯 Project Overview
The goal is to build, compare and deploy a road damage detector. Instead of saying *what* an image is, the model says **where** each damage is and **which type** it is, by drawing a box around it.

What was done:
- Trained and compared **3 YOLO models** under identical settings.
- Cleaned the dataset to remove leakage between train, validation and test.
- Evaluated with **mAP50, mAP50-95, precision, recall, inference speed and model size**.
- Picked the best model for deployment and built a **Gradio** web app.

---

## 📦 Dataset
- Road Damage dataset exported from Roboflow in YOLOv8 format.
- **7 classes:** `alligator`, `block`, `crack`, `edge`, `longitudinal`, `pothole`, `transverse`.
- Final split after cleaning: **4591 train / 1123 validation / 592 test** images.

| Class | Train instances | Val instances | Test instances |
|---|---|---|---|
| pothole | 5534 | 1546 | 909 |
| alligator | 4234 | 993 | 577 |
| longitudinal | 1892 | 537 | 311 |
| crack | 1704 | 226 | 93 |
| transverse | 394 | 113 | 67 |
| block | 305 | 86 | 47 |
| edge | 19 | 1 | 0 |

The `edge` class has only 19 training examples, so it cannot be learned. It is **excluded from the reported mAP** (mean over the remaining 6 classes).

---

## 🧹 Data Cleaning
1. **Leakage between splits.** The Roboflow export contains augmented copies of the same source image. Any training image that shares its source image with the validation or test set was removed from training.
2. **Near-duplicate check.** A perceptual hash (16x16 average hash) was used to find test images almost identical to training images. 89 of 681 test images were removed, leaving a **clean test set of 592 images**. Scores did not drop after this step, so the test result is not inflated by duplicates.
3. **Mixed label files.** A few label files mix box rows with polygon rows (5 in train, 4 in validation, 0 in test). YOLO ignores those images.

---

## 🏋️ Models and Training Setup
All models start from COCO-pretrained weights and use exactly the same settings, so the comparison is fair.

| Setting | Value |
|---|---|
| Epochs | 40 |
| Image size | 640 |
| Batch size | 16 |
| Early stopping patience | 15 |
| Seed | 42 |
| GPU | Tesla T4 (Google Colab) |

Models: **YOLO11n**, **YOLO11s**, **YOLOv8s**.

---

## 📊 Results

### Model comparison
| Model | mAP50 (val) | mAP50-95 (val) | mAP50 (test) | mAP50-95 (test) | Precision | Recall | Inference | Size | Params |
|---|---|---|---|---|---|---|---|---|---|
| **YOLO11n** | 0.471 | 0.267 | 0.595 | 0.372 | 0.595 | 0.572 | 4.3 ms (~133 FPS) | **5.5 MB** | 2.6M |
| YOLO11s | 0.481 | 0.280 | 0.598 | 0.373 | 0.601 | 0.578 | 9.4 ms (~85 FPS) | 19.2 MB | 9.4M |
| YOLOv8s | 0.492 | 0.283 | 0.595 | 0.365 | 0.666 | 0.548 | 9.7 ms (~79 FPS) | 22.5 MB | 11.1M |

- Test metrics are on the **clean test set (592 images)**; validation metrics are over 6 classes (no `edge`).
- Speed was measured on a Tesla T4 (batch 16). FPS includes pre- and post-processing.
- mAP50 uses an IoU threshold of 0.5; mAP50-95 averages IoU thresholds from 0.5 to 0.95, so both reflect localization quality (IoU).

### Choice for deployment: YOLO11n
The three models are within about one point of each other in mAP. **YOLO11n is about 4x smaller and about 2x faster**, so it is the most efficient to deploy.

### Per-class results (YOLO11n, clean test set)
| Class | mAP50 | mAP50-95 |
|---|---|---|
| crack | 0.890 | 0.681 |
| block | 0.842 | 0.638 |
| pothole | 0.671 | 0.362 |
| transverse | 0.558 | 0.294 |
| alligator | 0.352 | 0.165 |
| longitudinal | 0.257 | 0.090 |

`alligator` and `longitudinal` are the weakest classes in every model (recall around 0.25 to 0.30). Alligator cracks cover large areas with unclear boundaries, and the three crack types visually overlap, so the labels are ambiguous. The limit here looks like the data, not the model size.

---

## 🔧 Hyperparameter Tuning
Four short runs on YOLO11n (15 epochs, 30% of the training data), scored on validation (mAP50 / mAP50-95):

| Run | Change | mAP50 | mAP50-95 |
|---|---|---|---|
| t_base | defaults | 0.154 | 0.065 |
| t_lr005 | lr0 = 0.005 | 0.154 | 0.065 |
| t_img800 | image size 800 | 0.134 | 0.055 |
| t_aug | stronger augmentation | 0.157 | 0.068 |

Findings:
- Differences were within noise, so default settings were kept.
- `lr0` had **no effect**: in short runs Ultralytics picks the optimizer automatically (AdamW) and ignores `lr0`. Setting `optimizer="SGD"` explicitly is needed for a valid learning-rate comparison.
- A 800 px image size learned more slowly at this small budget.

---

## 🧪 Experiment: Merging Confusing Classes
Class size alone did not explain performance (`block` has only 305 training instances but scores 0.84, while `alligator` has 4234 and scores 0.35). The more likely problem is that the crack types look alike. So a second experiment merged them:

| New class | Original classes |
|---|---|
| `linear_crack` | crack + longitudinal + transverse + edge |
| `alligator` | alligator |
| `pothole` | pothole |
| `block` | block |

YOLO11n was retrained from COCO weights with the same settings (40 epochs, image size 640, batch 16, seed 42). While rebuilding the labels, polygon rows were converted to bounding boxes, which recovers the 9 label files that YOLO had been ignoring.

| Setup | Classes | mAP50 (val) | mAP50-95 (val) | mAP50 (test) | mAP50-95 (test) | Precision | Recall |
|---|---|---|---|---|---|---|---|
| YOLO11n (original) | 7 (mean over 6) | 0.471 | 0.267 | 0.595 | 0.372 | 0.595 | 0.572 |
| YOLO11n (merged) | 4 | 0.525 | 0.309 | 0.579 | 0.352 | 0.659 | 0.521 |

**Reading the result**
- The two rows are **not directly comparable**: the merged task has fewer classes and is easier by construction.
- Merging improved the larger validation set (+5.4 mAP50, +4.2 mAP50-95) and raised precision, but did **not** improve the test set (-1.6 mAP50, -2.0 mAP50-95), and recall dropped.
- A likely reason: in the 7-class setup the `crack` class scored 0.89 on the test set, but it has only 67 test instances, which flattered the original test score. In the merged setup, `linear_crack` covers 416 test instances, so the test score is based on more data.
- Conclusion: confusion between crack types is part of the problem, but not the only limit. Merging costs detail (the model can no longer separate longitudinal from transverse cracks) for a gain that is clear only on validation, so the 7-class model was kept as the main model.

---

## 🧗 Challenges and Lessons Learned
- **Data leakage.** The first check showed augmented copies of the same image across splits. Fixing it was more important than any model change.
- **Validation vs test gap.** Validation mAP50 (0.41 to 0.42 with `edge`) looked much lower than test (0.59). Two causes: the `edge` class scored 0 on validation, and the `crack` class scored 0.25 on validation but 0.88 on test (223 validation instances vs only 67 in test). After excluding `edge` and removing near-duplicates, the gap shrank but did not vanish, so both numbers are reported.
- **Fair metric.** `edge` was excluded from the mean because 19 training examples cannot be learned.
- **Misleading tuning.** Two runs with different learning rates gave identical scores, which exposed that `lr0` was being ignored (see Hyperparameter Tuning).
- **GPU limits.** Each 40-epoch run took 1 to 2 hours on a free T4, which limited how many experiments could be run.
- **Low confidence on some images.** On a close-up alligator-crack image, the model drew many overlapping boxes with confidence 0.05 to 0.12. This is a real failure case that matches the low alligator recall.

---

## 🖥️ Deployment App
A **Gradio** app serves the YOLO11n model.

- Upload a road image and get boxes around detected damage.
- Adjustable **confidence threshold** (default 0.25).
- A summary of how many instances of each damage type were found.
- Overlapping boxes are merged (`iou=0.4`).

Tip: use pothole, block and crack images to see high confidence. Alligator and longitudinal cracks are harder for the model.

---

## 🧭 Step-by-Step Guide

### A. Reproduce the training (Google Colab)
1. **Get the dataset.** Download the Road Damage dataset from Roboflow in **YOLOv8 format**. You get a zip named `road-damage.v1i.yolov8.zip`. (The dataset is not included in this repository.)
2. **Put it on Google Drive.** Create a folder `road_damage_runs` in your Drive and place the zip inside it.
3. **Open the notebook.** Open `notebooks/yolo08s.ipynb` in Google Colab and choose **Runtime → Change runtime type → T4 GPU**.
4. **Run the cells in order, one by one** (Shift + Enter):
   - Setup (installs the libraries and connects Drive; allow access when asked).
   - Data loading and leakage cleaning (should print `4591 1123 681`).
   - Settings, then the clean test set (should print `592`).
   - Then the training sections: YOLOv8s, YOLO11n, tuning, and the merged-classes experiment.
5. **Read the results.** Each training cell ends by printing a `RESULT` block with mAP50, mAP50-95, precision, recall, speed and model size.
6. **Find your model.** Weights are saved on Drive at `road_damage_runs/<run_name>/weights/best.pt` (for example `yolo11n_run`).
7. **YOLO11s and the data checks** are in `notebooks/yolo11s.ipynb`.

> Each 40-epoch run takes about 1 to 2 hours on a free T4. Do not close the Colab tab while it trains. Results are saved on Drive after every epoch, so a disconnected run can be resumed.

### B. Run the app on your computer (Windows)
1. Create a folder (for example `road_app`) and put these files in it: `app.py`, `requirements.txt`, `best.pt`.
2. Open the folder in File Explorer, click the address bar, type `cmd` and press Enter. A terminal opens inside the folder.
3. Install the libraries (this takes a few minutes the first time):
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   python app.py
   ```
5. Open **http://127.0.0.1:7860** in your browser. Keep the terminal window open while you use the app; closing it stops the app.
6. Upload a road image, adjust the confidence threshold if needed, and press **Detect**.

If `pip` is not recognized, use `python -m pip install -r requirements.txt` instead.

### C. Publish the app online (Hugging Face Spaces)
GitHub only stores the code; it cannot run the app. To get a public link:
1. Create a free account on huggingface.co.
2. Click **New → Space**. Choose SDK **Gradio**, hardware **CPU basic (free)**, and **Public**.
3. Upload `app.py`, `requirements.txt`, `best.pt`, and a `README.md` that starts with this header:
   ```
   ---
   title: Road Damage Detection
   sdk: gradio
   app_file: app.py
   ---
   ```
4. Wait a couple of minutes for the build. The Space gets a public link.
5. Paste that link into the **Live demo** line at the top of this README.

### D. How to read the app output
- **Boxes** are drawn around each detected damage, labeled with the class and a confidence score.
- **Lower the confidence threshold** (for example to 0.15) if nothing is detected on an image that clearly has damage.
- **Pothole, block and crack** are detected with the highest confidence. **Alligator and longitudinal** cracks are harder and usually get low confidence.

---

## ⚠️ Limitations and Future Work
- Absolute mAP is moderate (about 0.6 mAP50 and 0.37 mAP50-95 on the clean test set).
- Hyperparameter tuning was limited to short runs because of GPU limits.
- A **medium-size model was not trained** (GPU limits); the nano and small results are already close, so a large gain is not expected.
- The `edge` class cannot be learned with 19 examples.
- Merging the crack classes was tried (see the experiment above) and gave a gain on validation only.
- Possible improvements: longer training (80+ epochs), relabeling the confusing crack classes, and a mean-IoU metric of matched detections.

---

## 📁 Repository Structure
```
road-damage-detection/
├── app.py                 # Gradio app
├── best.pt                # Trained YOLO11n weights
├── requirements.txt       # Dependencies
├── README.md
├── notebooks/
│   ├── yolo08s.ipynb      # YOLOv8s, YOLO11n, tuning and merged-classes experiment
│   └── yolo11s.ipynb      # YOLO11s training, evaluation and data checks
└── screenshots/           # App screenshots (optional)
```

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python app.py
```
Then open **http://127.0.0.1:7860** in your browser. `best.pt` must be in the same folder as `app.py`.

---

> For educational purposes only.
