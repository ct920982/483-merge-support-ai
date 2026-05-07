import cv2
import csv
import numpy as np
from ultralytics import YOLO

# other .py files
import dashcam as dc

# model configurations
MODEL = "yolo26x.pt" # https://docs.ultralytics.com/models/yolo26/#key-features
CONF = 0.4
CLASSES = [0, 1, 2, 3, 5, 7, 9] # https://docs.ultralytics.com/datasets/detect/coco8/#introduction

class UnprotectedLeftTurnFeatureExtraction:
    def __init__(self):
        self.model = YOLO(MODEL)
        self.traf_light_memory = {}
        self.prev_dists = {}

        # TODO: make log file name dynamic
        # TODO: add is_safe
        self.csv_file = open("log.csv", "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["frame", "track_id", "class", "dist", "delta_dist", "light_state"])

    def get_traffic_light_state(self, frame, x1, y1, x2, y2):
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
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

    def estimate_distance(self, y2):
        delta_y = y2 - dc.HEIGHT / 2
        angle = np.arctan(delta_y / dc.FOCAL_LEN) + abs(dc.PITCH_RAD)

        if angle <= 0:
            return None
        return round(dc.HEIGHT / np.tan(angle), 1)

    def run_model(self, input_path, output_path):
        print("Running model...") # just for sanity check

        capture = cv2.VideoCapture(input_path)
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                                dc.FPS, dc.FRAME_SIZE)
        
        frame_count = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            results   = self.model.track(frame, persist=True, conf=CONF, classes=CLASSES,
                                    tracker="bytetrack.yaml", verbose=False)
            annotated = results[0].plot(labels=False, conf=False) # turn off yolo id and precision label

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    class_id = int(box.cls[0])

                    dist = self.estimate_distance(y2)
                    if dist is None:
                        continue

                    delta_dist = round(self.prev_dists.get(track_id, dist) - dist, 2)
                    self.prev_dists[track_id] = dist

                    # traffic lights
                    state = "N/A"
                    color = (255, 255, 255)

                    if class_id == 9: 
                        detected_state = self.get_traffic_light_state(frame, x1, y1, x2, y2)
                        if detected_state == "unkown":
                            state = self.traf_light_memory.get(track_id, "unkown")
                        else:
                            state = detected_state
                            self.traf_light_memory[track_id] = state
                        
                        state_colors = {"red": (0,0,255), "yellow": (0,255,255), "green": (0,255,0)}
                        color = state_colors.get(state, (255, 255, 255))
                    
                    # check if object is center left aka left lane
                    is_oncoming = (dc.FRAME_WIDTH * 0.3 < (x1 + x2) / 2 < dc.FRAME_WIDTH * 0,5)
                    text_color =  (0, 165, 255) if is_oncoming and delta_dist > 0 else color

                    label = f"ID{track_id}: {dist}m"
                    if state != "N/A": label += f" [{state}]"
                    cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 2)
                    
                    self.writer.writerow([frame_count, track_id, class_id, dist, delta_dist, state])
            
            output.write(annotated)
            frame_count += 1

        capture.release()
        output.release()
        self.csv_file.close()
        print(f"Success!!! Saved to {output_path}")
