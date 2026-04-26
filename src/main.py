import cv2
from ultralytics import YOLO

# input/output paths need to be implemented locally since /data is not pushed
INPUT_PATH = '../data/raw/Video Apr 14 2026, 12 05 06 PM.mp4'
OUTPUT_PATH = '../data/processed/vid.mp4'
MODEL = 'yolo11n.pt' # nano model for now for fast processing
CONF = 0.4
CLASSES = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck

# This is the dashcam specs, should never change
FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
FPS = 25
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

model = YOLO(MODEL)
capture = cv2.VideoCapture(INPUT_PATH)
output = cv2.VideoWriter(OUTPUT_PATH, FOURCC, FPS, FRAME_SIZE)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results   = model.track(frame, persist=True, conf=CONF, classes=CLASSES,
                             tracker="bytetrack.yaml", verbose=False)
    annotated = results[0].plot()
    output.write(annotated)

capture.release()
output.release()
print(f"Success!!! Saved to {OUTPUT_PATH}")