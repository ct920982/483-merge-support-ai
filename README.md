# CPSC 483 Introduction to Machine Learning

## Final Project: Unprotected Left Turn Merge Support AI

### Project Overview

This project analyzes dashcam video to detect traffic objects, estimate distance, track movement, detect traffic light color, and decide whether an unprotected left turn is safe.

The pipeline uses YOLO object tracking, feature extraction, labeled CSV logs, trained classifiers, and model evaluation.

---

## Folder Setup

Make sure the project is organized like this:

```text
project/
├── data/
│   ├── raw/              # input MP4 videos
│   ├── processed/        # output annotated MP4 videos
│   ├── logs/             # generated CSV feature logs
│   └── labeled_logs/     # manually labeled CSV files
├── models/               # YOLO model + saved classifiers
├── src/
│   ├── gui.py
│   ├── feature_extraction.py
│   ├── dashcam.py
│   ├── train_classifiers.ipynb
│   └── evaluate_models.ipynb
```

---

## Required Libraries

Make sure the following is installed globally:
- python3
- pip
- venv
- tkinter 

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### 1. Add Raw Video

Place the input dashcam video in:

```text
../data/raw/
```

From https://www.dropbox.com/scl/fo/nnabh3xd15apn6069kfdw/AFNPmfpu_KLq8Yf64i2jf1o?rlkey=6vtre7y7b0l21n989spazqqhq&st=kdxvl8ka&dl=0

> Professor sample will have videos included

---

### 2. Run Feature Extraction

Inside `src/` run:

```bash
python gui.py
```

A file picker will open. Select the input video.

This will generate:

```text
../data/logs/video_name.csv
../data/processed/video_name.mp4
```

The CSV contains extracted features:

```text
frame, track_id, class, dist, delta_dist, light_state, is_safe, is_oncoming
```

---

### 3. Manually Label the Data

Open the generated CSV file from:

```text
../data/logs/
```

Review and correct the `is_safe` column manually. Use associate processed videos to help.

After labeling, move the file to:

```text
../data/labeled_logs/
```

> Professor sample will have labeled csv files included

---

### 4. Train the Classifiers

Open and run:

```text
train_classifiers.ipynb
```

This notebook:

1. Loads all labeled CSV files.
2. Preprocesses the features.
3. Trains multiple classifiers.
4. Prints classification reports.
5. Saves trained models into:

```text
../models/
```

---

### 5. Evaluate the Models

Open and run:

```text
evaluate_models.ipynb
```

This notebook:

1. Loads the labeled data.
2. Applies the same preprocessing as training.
3. Loads the saved models.
4. Runs predictions.
5. Prints accuracy, precision, recall, F1-score, and ROC-AUC.
6. Displays confusion matrices and ROC curves.

---

## Important Notes

- Do not change `dashcam.py` unless the camera setup changes.
- `is_safe` must be manually reviewed before training.
- The project depends on the folder paths staying consistent.
- The processed MP4 is only for visual inspection.
- The labeled CSV files are used for training and evaluation.

---