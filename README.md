# CPSC 483 Introduction to Machine Learning  
## Final Project: Unprotected Left Turn Merge Support AI

---

## Pre-Requestitions
Make sure the following is installed globally:
- python3
- pip
- venv
- tkinter  

if not run:  
```bash
sudo apt install python3 python3-pip python3-venv tkinter
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alex-bo7/483-merge-support-ai.git
   cd 483-merge-support-ai/
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

Raw dashcam footage is not included in this repository due to file size. Videos are stored externally.

**To set up the dataset:**

1. Download the dashcam `.mp4` files from the shared Dropbox folder: `https://www.dropbox.com/scl/fo/nnabh3xd15apn6069kfdw/AFNPmfpu_KLq8Yf64i2jf1o?rlkey=6vtre7y7b0l21n989spazqqhq&st=kdxvl8ka&dl=0`
2. Place all `.mp4` files into `data/raw/`  
**OR**  
labeled `.csv` files are provided in `data/logs/` (to skip step 1 in How to Run section)

**CSV format** (produced by `feature_extraction.py` and manually labeled):

| Column | Description |
|---|---|
| `frame` | Frame number |
| `track_id` | Object tracking ID |
| `class` | COCO class ID |
| `dist` | Estimated distance in meters |
| `delta_dist` | Change in distance from previous frame |
| `light_state` | Traffic light color (`red`, `yellow`, `green`, `N/A`) |
| `is_safe` | Binary label — **manually set** (1 = safe, 0 = unsafe) |

> **Note:** After running feature extraction, the `is_safe` column defaults to `0`. Labels must be manually reviewed and corrected in the CSV before training.

---

## How to Run

### Step 1: Feature Extraction

Run the GUI to select a dashcam video. The script will process the video with YOLO, annotate it, and save the feature CSV.

```bash
python src/gui.py
```

- A file dialog will open, select an `.mp4` file from `data/raw/`
- Annotated video is saved to `data/processed/`
- Feature CSV is saved to `data/logs/`
- **Manually update the `is_safe` column** in the CSV before proceeding

### Step 2: Model Training & Evaluation

Open and run the Jupyter notebook:

```bash
jupyter notebook src/ml_models_evaluation.ipynb
```

- Update `csv_file_name` in the notebook to point to your labeled CSV
- The notebook trains and evaluates three classifiers:
  - Logistic Regression (baseline)
  - Decision Tree
  - Random Forest
- Outputs include accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves

---

## Dashcam Configuration

Camera constants are defined in `dashcam.py` and should not be changed unless a different camera is used.

| Parameter | Value | Description |
|---|---|---|
| Resolution | 3840 × 2160 | 4K |
| FPS | 25 | Frames per second |
| Mount height | 1.245 m | Camera height above ground |
| Pitch | −5.0° | Downward tilt angle |
| Focal length | 1578 px | Derived from 160° FOV at 4K |
