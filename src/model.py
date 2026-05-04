import cv2
from ultralytics import YOLO
import numpy as np

# other .py files
import dashcam as dc

# NOTES: need to fine tune dist?

MODEL = 'yolo11n.pt' # nano model for now for fast processing
CONF = 0.4
CLASSES = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck

def estimate_distance(y2):
    delta_y = y2 - dc.HEIGHT / 2
    angle = np.arctan(delta_y / dc.FOCAL_LEN) + abs(dc.PITCH_RAD)

    if angle <= 0:
        return None
    return round(dc.HEIGHT / np.tan(angle), 1)

def run_model(input_path, output_path):
    print("Running model...") # just for sanity check

    model = YOLO(MODEL)
    capture = cv2.VideoCapture(input_path)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                            dc.FPS, dc.FRAME_SIZE)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results   = model.track(frame, persist=True, conf=CONF, classes=CLASSES,
                                tracker="bytetrack.yaml", verbose=False)
        annotated = results[0].plot(labels=False, conf=False) # turn off yolo id and precision labels

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else -1
                dist = estimate_distance(y2)
                if dist is None:
                    continue
                cv2.putText(annotated, f"ID{track_id}: {dist}m",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) # label
        
        output.write(annotated)

    capture.release()
    output.release()
    print(f"Success!!! Saved to {output_path}")

if __name__ == "__main__": # so it doesn't run when imported, only when called
    run_model()