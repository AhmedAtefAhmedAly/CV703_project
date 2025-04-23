import os
import math
import json
import glob
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R
from bpc.inference.utils.camera_utils import load_camera_params

# Make sure to set the OpenGL platform before importing pyrender.
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender


def compute_2d_center(K, R_mat, t):
    """
    Projects the 3D translation t (of shape (3,1)) into 2D via K.
    Returns (u, v) or None if behind the camera.
    """
    if t[2, 0] <= 0:
        return None
    uv = K @ t
    if uv[2, 0] == 0:
        return None
    uv /= uv[2, 0]
    return uv[0, 0], uv[1, 0]


def letterbox_preserving_aspect_ratio(img, target_size=256, fill_color=(255, 255, 255)):
    """
    Resizes and pads an image to the target_size while preserving aspect ratio.
    If the image is grayscale (2D), a single-channel canvas is created.
    """
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Determine if the image is color (3 channels) or grayscale.
    if len(img.shape) == 3 and img.shape[2] == 3:
        canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    else:
        # For grayscale, fill_color can be a scalar (use first value if tuple provided).
        fill_scalar = fill_color if isinstance(fill_color, int) else fill_color[0]
        canvas = np.full((target_size, target_size), fill_scalar, dtype=np.uint8)
    
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    # Place resized image on canvas.
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


def matrix_to_euler_xyz(R_mat):
    """
    Convert a 3x3 rotation matrix to Euler angles (x, y, z) in radians,
    assuming the rotation matrix was built via: R = Rx(x) * Ry(y) * Rz(z).
    """
    sy = R_mat[0, 2]
    eps = 1e-7
    if abs(abs(sy) - 1.0) > eps:
        y = math.asin(sy)
        x = math.atan2(-R_mat[1, 2], R_mat[2, 2])
        z = math.atan2(-R_mat[0, 1], R_mat[0, 0])
    else:
        y = math.pi / 2 if sy > 0 else -math.pi / 2
        x = math.atan2(R_mat[2, 1], R_mat[1, 1])
        z = 0.0
    return (x, y, z)


def euler_to_quat(euler_angles):
    """
    Convert Euler angles (Rx, Ry, Rz) in radians to quaternion [x, y, z, w].
    """
    r = R.from_euler('xyz', euler_angles, degrees=False)
    return r.as_quat()


def euler_to_6d(euler_angles):
    """
    Convert Euler angles to a 6D rotation representation.
    Compute the rotation matrix and take its first two columns flattened.
    """
    r = R.from_euler('xyz', euler_angles, degrees=False)
    R_mat = r.as_matrix()  # shape (3,3)
    return np.concatenate([R_mat[:, 0], R_mat[:, 1]])

def get_modality_path(rgb_path, modality_prefix):
    """
    Given an rgb_path (e.g., .../rgb_cam2/000002.jpg), replace "rgb_" with the given modality_prefix (e.g., "depth_" or "polar_"),
    and return the path if it exists. If the resulting file doesn't exist, try switching the extension between .jpg and .png.
    """
    base_path = rgb_path.replace("rgb_", modality_prefix)
    if os.path.exists(base_path):
        return base_path
    else:
        root, ext = os.path.splitext(base_path)
        alt_ext = ".png" if ext.lower() == ".jpg" else ".jpg"
        alt_path = root + alt_ext
        if os.path.exists(alt_path):
            return alt_path
    return None


class BOPSingleObjDataset(Dataset):
    """
    A dataset for a single object ID from BOP data.
    
    For training (split == "train" and augment==True), each __getitem__ returns a tuple:
        (orig_img_t, aug_img_t, label_dict, meta)
    """
    def __init__(self,
                 root_dir,
                 scene_ids,
                 cam_ids,
                 target_obj_id,
                 target_size=256,
                 augment=False,
                 split="train",
                 max_per_scene=None,
                 train_ratio=0.8,  # Default split ratio: 80% train, 20% val.
                 seed=42,
                 use_real_val=False  # New flag: if True, try to use the real validation dataset.
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.cam_ids = cam_ids
        self.obj_id = target_obj_id
        self.target_size = target_size
        self.augment = augment
        self.split = split.lower()  # "train" or "val"
        self.max_per_scene = max_per_scene
        self.train_ratio = train_ratio
        self.samples = []
        random.seed(seed)

        # Choose dataset path based on split and flag.
        if self.split == "val" and use_real_val:
            real_val_path = os.path.join(root_dir, "val")
            if os.path.exists(real_val_path):
                dataset_path = real_val_path
            else:
                print(f"[WARNING] Real validation directory {real_val_path} not found. Falling back to train_pbr.")
                train_pbr_path = os.path.join(root_dir, "train_pbr")
                if os.path.exists(train_pbr_path):
                    dataset_path = train_pbr_path
                else:
                    raise FileNotFoundError(f"Directory '{train_pbr_path}' not found in {root_dir}")
        else:
            train_pbr_path = os.path.join(root_dir, "train_pbr")
            if os.path.exists(train_pbr_path):
                dataset_path = train_pbr_path
            else:
                raise FileNotFoundError(f"Directory '{train_pbr_path}' not found in {root_dir}")

        # Gather all samples.
        all_samples = []
        for sid in scene_ids:
            scene_path = os.path.join(dataset_path, sid)
            scene_count = 0
            for cam_id in self.cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file  = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                rgb_dir   = os.path.join(scene_path, f"rgb_{cam_id}")
                if not all(os.path.exists(f) for f in [info_file, pose_file, cam_file, rgb_dir]):
                    continue
                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(cam_file, "r") as f3:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json  = json.load(f3)
                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue
                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                    
                    img_name_jpg = os.path.join(rgb_dir, f"{im_id:06d}.jpg")
                    img_name_png = os.path.join(rgb_dir, f"{im_id:06d}.png")
                    
                    # Check if JPG exists, otherwise use PNG.
                    if os.path.exists(img_name_jpg):
                        img_path = img_name_jpg
                    elif os.path.exists(img_name_png):
                        img_path = img_name_png
                    else:
                        continue  # Skip if neither format is found.
                    
                    # Loop through all object instances in the image.
                    for inf, pos in zip(info_json[im_id_s], pose_json[im_id_s]):
                        if pos["obj_id"] != self.obj_id:
                            continue
                    
                        # Use bbox_visib for the visible part of the object.
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue
                    
                        # Retrieve additional BOP challenge metrics if available.
                        visib_fract = inf.get("visib_fract", 1.0)
                        px_count_all = inf.get("px_count_all", w_ * h_)
                        px_count_valid = inf.get("px_count_valid", px_count_all)
                    
                        # Apply filtering thresholds.
                        if visib_fract < 0.1 or px_count_valid < 1000:
                            continue
                    
                        R_mat = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)
                        all_samples.append({
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "img_path": img_path,
                            "K": K,
                            "R": R_mat,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_],
                            "visib_fract": visib_fract,
                            "px_count_all": px_count_all,
                            "px_count_valid": px_count_valid
                        })
                scene_count += 1
                if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                    break

        # ---- SPLITTING LOGIC ----
        # When using real validation data, assume the samples are pre-defined.
        # Otherwise, split the synthetic data using train_ratio.
        from collections import defaultdict
        groups = defaultdict(list)
        for s in all_samples:
            key = (s["scene_id"], s["cam_id"])
            groups[key].append(s)
        final_samples = []
        for group_samples in groups.values():
            random.shuffle(group_samples)
            n_total = len(group_samples)
            n_train = int(round(self.train_ratio * n_total))
            if self.split == "train":
                selected = group_samples[:n_train]
            else:  # self.split == "val"
                selected = group_samples[n_train:]
            final_samples.extend(selected)
        self.samples = final_samples

        print(f"[INFO] BOPSingleObjDataset(split={self.split}, augment={self.augment}): total={len(self.samples)} samples.")


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        data = self.samples[idx]
        img_path = data["img_path"]
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise IOError(f"Cannot read {img_path}")

        # Load corresponding depth image.
        depth_path = get_modality_path(img_path, "depth_")
        if depth_path is None:
            depth = np.zeros(bgr.shape[:2], dtype=np.float32)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                depth = np.zeros(bgr.shape[:2], dtype=np.float32)
            else:
                depth = depth.astype(np.float32) / np.max(depth)

        # Load corresponding polarization image.
        polar_path = get_modality_path(img_path, "polar_")
        if polar_path is None:
            polarization = np.zeros(bgr.shape[:2], dtype=np.float32)
        else:
            polarization = cv2.imread(polar_path, cv2.IMREAD_GRAYSCALE)
            if polarization is None:
                polarization = np.zeros(bgr.shape[:2], dtype=np.float32)
            else:
                polarization = polarization.astype(np.float32) / 255.0

        # ===== Existing processing for RGB =====
        K = data["K"]
        R_mat = data["R"]
        t = data["t"]
        x, y, w, h = map(int, data["bbox_visib"])
        H_img, W_img = bgr.shape[:2]

        # Compute the original (non-augmented) crop for RGB.
        orig_crop = bgr[y:y+h, x:x+w]
        if orig_crop.size == 0:
            raise RuntimeError("Empty crop for original image")
        orig_letter_img, orig_scale, orig_dx, orig_dy = letterbox_preserving_aspect_ratio(orig_crop, target_size=self.target_size)
        orig_letter_img_c = np.ascontiguousarray(orig_letter_img, dtype=np.uint8)
        orig_img_t = torch.from_numpy(orig_letter_img_c).permute(2, 0, 1).float() / 255.0

        # ===== Process depth and polarization crops similarly =====
        # For simplicity, we apply the same crop as the RGB image.
        depth_crop = depth[y:y+h, x:x+w]
        depth_letter, _, _, _ = letterbox_preserving_aspect_ratio(depth_crop, target_size=self.target_size, fill_color=0)
        # depth_letter is still single-channel (H, W, 3) if letterbox_preserving_aspect_ratio returns a 3-channel image.
        # Ensure it's single-channel:
        if len(depth_letter.shape) == 3 and depth_letter.shape[2] == 3:
            depth_letter = cv2.cvtColor(depth_letter, cv2.COLOR_BGR2GRAY)
        depth_t = torch.from_numpy(depth_letter).unsqueeze(0).float()  # shape: [1, target_size, target_size]

        polar_crop = polarization[y:y+h, x:x+w]
        polar_letter, _, _, _ = letterbox_preserving_aspect_ratio(polar_crop, target_size=self.target_size, fill_color=0)
        if len(polar_letter.shape) == 3 and polar_letter.shape[2] == 3:
            polar_letter = cv2.cvtColor(polar_letter, cv2.COLOR_BGR2GRAY)
        polar_t = torch.from_numpy(polar_letter).unsqueeze(0).float()  # shape: [1, target_size, target_size]

        # Process augmentation for RGB if needed (same as before).
        aug_img_t = None
        if self.split == "train" and self.augment:
            # [Augmentation code remains the same...]
            scale_factor = 1.0 + 0.2 * random.random()
            aug_w = int(round(w * scale_factor))
            aug_h = int(round(h * scale_factor))
            max_shift_x = int(0.1 * w)
            max_shift_y = int(0.1 * h)
            shift_x = random.randint(-max_shift_x, max_shift_x)
            shift_y = random.randint(-max_shift_y, max_shift_y)
            aug_x = max(0, min(x - shift_x, W_img - 1))
            aug_y = max(0, min(y - shift_y, H_img - 1))
            aug_w = min(aug_w, W_img - aug_x)
            aug_h = min(aug_h, H_img - aug_y)
            aug_crop = bgr[aug_y:aug_y+aug_h, aug_x:aug_x+aug_w]
            if aug_crop.size == 0:
                raise RuntimeError("Empty crop for augmented image")
            aug_letter_img, aug_scale, aug_dx, aug_dy = letterbox_preserving_aspect_ratio(aug_crop, target_size=self.target_size)
            aug_letter_img_c = np.ascontiguousarray(aug_letter_img, dtype=np.uint8)
            aug_img_t = torch.from_numpy(aug_letter_img_c).permute(2, 0, 1).float() / 255.0
            jitter_transform = T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.05, hue=0.05)
            img_pil = TF.to_pil_image(aug_img_t)
            img_pil = jitter_transform(img_pil)
            aug_img_t = TF.to_tensor(img_pil)
            aug_img_t = TF.normalize(aug_img_t, mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        # Normalize the original image.
        orig_img_t = TF.normalize(orig_img_t, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        # ===== Compute rotation representations =====
        Rx, Ry, Rz = matrix_to_euler_xyz(R_mat)
        euler_angles = np.array([Rx, Ry, Rz], dtype=np.float32)
        quat = euler_to_quat(euler_angles)
        rep6d = euler_to_6d(euler_angles)
        label_dict = {
            "euler": np.array(euler_angles, dtype=np.float32),
            "quat":  np.array(quat, dtype=np.float32),
            "6d":    np.array(rep6d, dtype=np.float32),
            "R":     np.array(R_mat, dtype=np.float32)
        }
        for key in label_dict:
            label_dict[key] = torch.from_numpy(label_dict[key])
        
        meta = {
            "scene_id": data["scene_id"],
            "cam_id": data["cam_id"],
            "im_id": data["im_id"]
        }

        # ---- Extended return with additional modalities ----
        # You can choose to return a tuple or a dictionary.
        # Here, we return a tuple with extra elements: (orig_rgb, aug_rgb, depth, polarization, label_dict, meta)
        return (orig_img_t, aug_img_t, depth_t, polar_t, label_dict, meta)



def bop_collate_fn(batch):
    """
    Custom collate function that collates images, labels, additional modalities, and metadata.
    """
    orig_imgs, aug_imgs, depths, polars, labels, metas = [], [], [], [], [], []
    for sample in batch:
        orig, aug, depth, polar, lbl, meta = sample
        orig_imgs.append(orig)
        depths.append(depth)
        polars.append(polar)
        labels.append(lbl)
        metas.append(meta)
        if aug is not None:
            aug_imgs.append(aug)
    if len(aug_imgs) > 0:
        imgs_t = torch.cat([torch.stack(orig_imgs, dim=0), torch.stack(aug_imgs, dim=0)], dim=0)
        # For extra modalities, duplicate them to match the augmented set.
        depths = depths + depths
        polars = polars + polars
        labels = labels + labels
        metas = metas + metas
    else:
        imgs_t = torch.stack(orig_imgs, dim=0)
    # Instead of returning just the images, return a tuple with all modalities.
    # For simplicity, we return lists for depths and polars.
    return imgs_t, depths, polars, labels, metas

def render_mask(mesh, K, camera_pose, imsize, mesh_poses):
    K = K.copy()
    camera_pose = camera_pose.copy()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.background_color = np.array([0.0, 0.0, 0.0])
    for mesh_pose in mesh_poses:
        scene.add(mesh, pose=mesh_pose)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=10000)
    camera_pose[1, :] = -camera_pose[1, :]
    camera_pose[2, :] = -camera_pose[2, :]
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)
    light_direction = np.array([0, 0, -1])
    light_direction_world = camera_pose.copy()
    light_direction_world[:3, :3] = light_direction_world[:3, :3] @ light_direction
    light = pyrender.DirectionalLight(color=np.array([1.0, 0, 1.0]), intensity=5)
    scene.add(light, pose=light_direction_world)
    renderer = pyrender.OffscreenRenderer(*imsize)
    color, depth = renderer.render(scene)
    return color, depth


def load_gt_poses(scene_dir, scene_id, cam_ids, image_id, obj_id):
    gt_poses = []
    scene_path = os.path.join(scene_dir, scene_id)
    for cam_id in cam_ids[:1]:
        gt_path = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
        info_path = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
        if not os.path.exists(gt_path) or not os.path.exists(info_path):
            print(f"Missing GT files for {cam_id}")
            continue
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        with open(info_path, "r") as f:
            info_data = json.load(f)
        img_key = str(image_id)
        if img_key not in gt_data or img_key not in info_data:
            print(f"Image {image_id} not found in {cam_id}")
            continue
        objects = gt_data[img_key]
        bboxes = info_data[img_key]
        for obj, bbox in zip(objects, bboxes):
            if obj["obj_id"] != obj_id:
                continue
            rotation_matrix = np.array(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            translation = np.array(obj["cam_t_m2c"], dtype=np.float32)
            gt_poses.append(calc_pose_matrix(rotation_matrix, translation))
    return gt_poses


def calc_pose_matrix(R_mat, t):
    pose = np.eye(4)
    pose[:3, :3] = R_mat
    pose[:3, 3] = t
    return pose


class Capture:
    def __init__(self, images, Ks, RTs, obj_id, gt_poses=None):
        self.images = images
        self.Ks = Ks
        self.RTs = RTs
        print(gt_poses)
        if gt_poses:
            self.gt_poses = np.linalg.inv(RTs[0]) @ gt_poses

    @classmethod
    def from_dir(cls, scene_dir, cam_ids, image_id, obj_id):
        cam_params = load_camera_params(scene_dir, cam_ids)
        Ks = [cam_params[x]['K'][image_id] for x in cam_ids]
        Rs = [cam_params[x]['R'][image_id] for x in cam_ids]
        Ts = [cam_params[x]['t'][image_id] for x in cam_ids]
        RTs = [calc_pose_matrix(r, t) for r, t in zip(Rs, Ts)]
        image_paths = [glob.glob(os.path.join(scene_dir, f"rgb_{cam_id}", f"{image_id:06d}.*g"))[0] for cam_id in cam_ids]
        images = [cv2.imread(x) for x in image_paths]
        gt_poses = load_gt_poses(scene_dir, '', cam_ids, image_id, obj_id)
        return cls(images, Ks, RTs, obj_id, gt_poses)
