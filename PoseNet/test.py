# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Paths
# # model_path = "bpc/yolo/models/detection/obj_11/yolo11-detection-obj_11.pt"
# model_path = "models/detection/obj_11/yolo11-detection-obj_11.pt"
# image_path = "datasets/train_pbr/000005/rgb_cam1/000001.jpg"

# # Load YOLO model
# model = YOLO(model_path)

# # Load image using OpenCV
# img_bgr = cv2.imread(image_path)
# if img_bgr is None:
#     raise ValueError(f"Failed to load image: {image_path}")

# # Convert BGR to RGB (Matplotlib expects RGB format)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# # Run YOLO inference
# results = model(img_rgb)[0]

# # Draw detections directly on the image
# for box in results.boxes.xyxy:
#     x1, y1, x2, y2 = map(int, box[:4])
#     cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw bounding box in blue

# # Display the image using Matplotlib
# plt.figure(figsize=(8, 8))
# plt.imshow(img_rgb)
# plt.axis("off")  # Hide axis
# plt.title("YOLO Detection")
# plt.show()



import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import cv2
import glob
import json
import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)
from scipy.spatial.transform import Rotation as R

# Local imports from your project:
from bpc.inference.utils.camera_utils import load_camera_params
from bpc.inference.process_pose import PoseEstimator, PoseEstimatorParams
from bpc.utils.data_utils import Capture, render_mask
import bpc.utils.data_utils as du
import importlib

# Example paths and scene setup
scene_dir = "./datasets/ipd_val/val/000001/"
models_dir = ''
cam_ids = ["cam1", "cam2", "cam3"]
image_id = 3
obj_id = 14
obj_id_path = str(1000000 + obj_id)[1:]
ply_file = os.path.join(models_dir, f"obj_{obj_id_path}.ply")
obj = trimesh.load(ply_file)

# YOLO and pose model paths
# yolo_model_path = f'models/detection/detection/obj_{obj_id}/yolo11-detection-obj_{obj_id}.pt'
# pose_model_path = f'bpc/pose/pose_checkpoints/obj_{obj_id}/best_model.pth'

yolo_model_path = f'models/detection/obj_{obj_id}/yolo11-detection-obj_{obj_id}.pt'
pose_model_path = f'models/rot_models/rot_{obj_id}.pth'



# Configure the pose estimator
pose_params = PoseEstimatorParams(
    yolo_model_path=yolo_model_path,
    pose_model_path=pose_model_path,
    yolo_conf_thresh=0.01,
    # rotation_mode="quat"  # Using quaternion mode for the example.
)
pose_estimator = PoseEstimator(pose_params)

# Perform multi-camera detection, matching, and rotation inference
t_start = time.time()
capture = Capture.from_dir(scene_dir, cam_ids, image_id, obj_id)
detections = pose_estimator._detect(capture)
pose_predictions = pose_estimator._match(capture, detections)
pose_estimator._estimate_rotation(pose_predictions)
inference_time = time.time() - t_start
print("Inference took:", inference_time, "seconds")
print(pose_predictions)

# pose_predictions now contains the final poses for each matched object
# Evaluate these against ground-truth poses with your desired BOP metric.