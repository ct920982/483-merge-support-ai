import numpy as np

# These are the dashcam configs, should NEVER change

FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
FPS = 25

HEIGHT = 1.245 # meters
PITCH = -5.0 # basically the tilt
FOCAL_LEN = 1578 # pixels (4k res, 1.8mm(1241px), 160FOV)

PITCH_RAD = np.deg2rad(PITCH)