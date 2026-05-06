import cv2
from ultralytics import YOLO
import numpy as np

# other .py files
import dashcam as dc

# NOTES: need to fine tune dist?

MODEL = "yolo26x.pt" # https://docs.ultralytics.com/models/yolo26/#key-features
CONF = 0.4
CLASSES = [0, 1, 2, 3, 5, 7, 9] # https://docs.ultralytics.com/datasets/detect/coco8/#introduction

def get_traffic_light_state(frame, x1, y1, x2, y2):
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # HSV ranges for each color
    masks = {
        "red":    cv2.bitwise_or(
                    cv2.inRange(hsv, (0,100,100), (10,255,255)),
                    cv2.inRange(hsv, (160,100,100), (180,255,255))
                    ),
        "yellow": cv2.inRange(hsv, (15,100,100), (35,255,255)),
        "green":  cv2.inRange(hsv, (40,100,100), (90,255,255)),
    }

    scores = {color: cv2.countNonZero(mask) for color, mask in masks.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 50 else "unknown"

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
        annotated = results[0].plot(labels=False, conf=False) # turn off yolo id and precision label

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else -1
                dist = estimate_distance(y2)
                if dist is None:
                    continue

                class_id = int(box.cls[0])
                label = f"ID{track_id}: {dist}m"

                if class_id == 9:  # traffic light
                    state = get_traffic_light_state(frame, x1, y1, x2, y2)
                    state_colors = {"red": (0,0,255), "yellow": (0,255,255), "green": (0,255,0)}
                    color = state_colors.get(state, (255,255,255))
                    label = f"ID{track_id}: {dist}m [{state}]" # label is the color of the traffic light
                else:
                    color = (255, 255, 255)

                cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                
                #TODO: Every frame we need to write features into a csv file
        
        output.write(annotated)

    capture.release()
    output.release()
    print(f"Success!!! Saved to {output_path}")

if __name__ == "__main__": # so it doesn't run when imported, only when called
    run_model()