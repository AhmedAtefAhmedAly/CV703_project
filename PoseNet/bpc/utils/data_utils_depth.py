import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import json
import glob
import cv2
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T  # for color jitter, etc.
from bpc.inference.utils.camera_utils import load_camera_params
import trimesh

def compute_2d_center(K, R, t):
    """
    Projects the 3D point 't' (the object translation) into 2D via K.
    Returns (u, v) or None if behind camera.
    """
    if t[2, 0] <= 0:
        return None
    uv = K @ t
    if uv[2, 0] == 0:
        return None
    uv /= uv[2, 0]
    return uv[0, 0], uv[1, 0]


def letterbox_preserving_aspect_ratio(img, target_size=256, fill_color=(255, 255, 255)):
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, img.shape[-1]), fill_color[:img.shape[-1]], dtype=np.uint8)
    if len(resized.shape) == 2:  # If OpenCV removed the channel dim
        resized = resized[:, :, np.newaxis]  # Add back channel dim
    # print("\n After Resize", resized.shape, canvas.shape)

    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


def matrix_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        # Fallback
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z


class BOPSingleObjDataset(Dataset):
    """
    Reads BOP data for a single obj_id from multiple scenes & cameras.
    Then splits 80/20 for train/val *per scene/cam*.

    Returns (img_t, label_5d, meta).
    label_5d => [Rx, Ry, Rz, cx, cy].
    'augment' => random shift/scale bounding box + optional color jitter if training.
    'split' => "train" or "val" to pick that portion from each scene/cam.
    """

    def __init__(
        self,
        root_dir,
        scene_ids,
        cam_ids,
        target_obj_id,
        target_size=256,
        augment=False,
        split="train",
        max_per_scene=None,
        train_ratio=0.8,
        seed=42
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

        random.seed(seed)  # for reproducibility

        # gather all samples in a temp list
        all_samples = []

        # Loop over each scene
        train_pbr_path = os.path.join(root_dir, "train_pbr")
        for sid in scene_ids:
            scene_path = os.path.join(train_pbr_path, sid)
            scene_count = 0

            for cam_id in cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                rgb_dir = os.path.join(scene_path, f"rgb_{cam_id}")

                if not all(os.path.exists(f) for f in [info_file, pose_file, cam_file, rgb_dir]):
                    # skip if missing
                    continue

                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(cam_file, "r") as f3:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json = json.load(f3)

                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue

                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                    img_name = f"{im_id:06d}.jpg"
                    img_path = os.path.join(rgb_dir, img_name)
                    if not os.path.exists(img_path):
                        continue

                    # each image can contain multiple objects, pick the ones that match self.obj_id
                    # The same image can have multiple instances of the same obj, but let's gather them
                    for inf, pos in zip(info_json[im_id_s], pose_json[im_id_s]):
                        if pos["obj_id"] != self.obj_id:
                            continue
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue

                        R = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)

                        all_samples.append({
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "img_path": img_path,
                            "K": K,
                            "R": R,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_]
                        })

                scene_count += 1
                if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                    break

        obj_model = trimesh.load(os.path.join(root_dir, "models", "obj_"+str(target_obj_id).zfill(6)+".ply"))
        obj_vertices = np.array(obj_model.vertices)
        self.obj_vertices = torch.from_numpy(obj_vertices).to(torch.float32)
        # print("\n \n Bahey \n ", type(obj_vertices), obj_vertices.shape)

        # Now we want to split 80/20 (train/val) *per scene/cam*, not globally.
        # So let's group by (scene_id, cam_id) => list of samples.
        from collections import defaultdict
        groups = defaultdict(list)
        for s in all_samples:
            key = (s["scene_id"], s["cam_id"])
            groups[key].append(s)

        # Shuffle each group and split
        for key, group_samples in groups.items():
            random.shuffle(group_samples)
            n_total = len(group_samples)
            n_train = int(round(self.train_ratio * n_total))

            if self.split == "train":
                selected = group_samples[:n_train]
            else:  # "val"
                selected = group_samples[n_train:]

            self.samples.extend(selected)

        print(f"[INFO] BOPSingleObjDataset(split={self.split}, augment={self.augment}): total={len(self.samples)} samples.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        img_path = data["img_path"]
        bgr = cv2.imread(img_path)
        depth = cv2.imread(img_path.replace("rgb", "depth").replace("jpg", "png"), cv2.IMREAD_UNCHANGED)
        aolp = cv2.imread(img_path.replace("rgb", "aolp").replace("jpg", "png"), cv2.IMREAD_UNCHANGED)
        dolp = cv2.imread(img_path.replace("rgb", "dolp").replace("jpg", "png"), cv2.IMREAD_UNCHANGED)

        depth = np.expand_dims(depth, axis=-1)
        # print(depth.shape, aolp.shape, dolp.shape)

        if bgr is None:
            raise IOError(f"Cannot read {img_path}")

        K = data["K"]
        R = data["R"]
        t = data["t"]
        x, y, w, h = map(int, data["bbox_visib"])

        H_img, W_img = bgr.shape[:2]

        # BBox augment if self.augment and self.split=="train"
        # This is your first type of augmentation.
        if self.augment and self.split == "train":
            scale_factor = 1.0 + 0.2 * random.random()  # up to +20%
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))

            max_shift_x = int(0.1 * w)
            max_shift_y = int(0.1 * h)
            shift_x = random.randint(-max_shift_x, max_shift_x)
            shift_y = random.randint(-max_shift_y, max_shift_y)

            x0 = x - shift_x
            y0 = y - shift_y
            x0 = max(0, min(x0, W_img - 1))
            y0 = max(0, min(y0, H_img - 1))
            new_w = min(new_w, W_img - x0)
            new_h = min(new_h, H_img - y0)
        else:
            x0, y0 = x, y
            new_w, new_h = w, h

        crop = bgr[y0:y0+new_h, x0:x0+new_w]
        depth_crop = depth[y0:y0+new_h, x0:x0+new_w]
        if crop.size == 0:
            raise RuntimeError("Empty crop => skip")

        letter_img, scale, dx, dy = letterbox_preserving_aspect_ratio(
            crop, target_size=self.target_size
        )
        # print("\n \n Before \n \n ", np.max(depth_crop))
        letter_depth, scale, dx, dy = letterbox_preserving_aspect_ratio(
            depth_crop, target_size=self.target_size
        )
        # print("\n \n After \n \n ", np.unique(letter_depth))

        uv = compute_2d_center(K, R, t)
        if uv is None:
            raise RuntimeError("Behind camera => skip")
        u_full, v_full = uv
        local_u = (u_full - x0)
        local_v = (v_full - y0)
        letter_x = local_u * scale + dx
        letter_y = local_v * scale + dy

        # Convert R => Euler
        Rx, Ry, Rz = matrix_to_euler_xyz(R)
        label_5d = np.array([Rx, Ry, Rz, letter_x, letter_y], dtype=np.float32)

        # Convert to tensor
        letter_img_c = np.ascontiguousarray(letter_img, dtype=np.uint8)
        img_t = torch.from_numpy(letter_img_c).permute(2, 0, 1).float() / 255.0

        letter_depth_c = np.ascontiguousarray(letter_depth)
        depth_t = torch.from_numpy(letter_depth_c).permute(2, 0, 1).float()

        # Apply color jitter if self.augment and split="train" as your second type of augmentation
        if self.augment and self.split == "train":
            # Example color jitter with random brightness/contrast/hue
            jitter_transform = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            # Convert to PIL for transform
            img_pil = TF.to_pil_image(img_t)
            img_pil = jitter_transform(img_pil)
            img_t = TF.to_tensor(img_pil)

        # Finally normalize (ImageNet style)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

        lbl_t = torch.from_numpy(label_5d)
        meta = {
            "scene_id": data["scene_id"],
            "cam_id": data["cam_id"],
            "im_id": data["im_id"]
        }
        return img_t, depth_t, lbl_t, self.obj_vertices, meta


def bop_collate_fn(batch):
    imgs, depths, labels, obj_vertices, metas = [], [], [], [], []
    for (img, depth, lbl, obj_vertice, meta) in batch:
        imgs.append(img)
        depths.append(depth)
        labels.append(lbl)
        obj_vertices.append(obj_vertice)
        metas.append(meta)
    imgs_t = torch.stack(imgs, dim=0)
    depths_t = torch.stack(depths, dim=0)
    labels_t = torch.stack(labels, dim=0)
    obj_vertices = torch.stack(obj_vertices, dim=0)
    return imgs_t, depths_t, labels_t, obj_vertices, metas



def render_mask(mesh, K, camera_pose, imsize, mesh_poses):
    """
    Render the mesh with the given camera and mesh pose on the full image,
    with a directional light from the camera's POV.
    """
    K = K.copy()
    camera_pose = camera_pose.copy()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.background_color = np.array([0.0, 0.0, 0.0]) # Set background to black

    for mesh_pose in mesh_poses:
        scene.add(mesh, pose=mesh_pose)

    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=10000)
    camera_pose[1, :] = -camera_pose[1, :]
    camera_pose[2, :] = -camera_pose[2, :]
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)

    # Add directional light from camera POV
    light_direction = np.array([0, 0, -1])  # Direction light points -Z axis in camera space.
    # Transform light direction to world space.
    light_direction_world = camera_pose.copy()
    light_direction_world[:3, :3] = light_direction_world[:3, :3] @ light_direction
    #The position of the light. We can put it at the camera's position.
    light = pyrender.DirectionalLight(
        color=np.array([1.0, 0, 1.0]), intensity=5)
    scene.add(light, pose=light_direction_world) #Light is at the origin in world space.

    renderer = pyrender.OffscreenRenderer(*imsize)
    color, depth = renderer.render(scene)
    return color, depth


def load_gt_poses(scene_dir, scene_id, cam_ids, image_id, obj_id):
    """Load all GT 6D poses and bounding boxes from scene_gt_camX.json and scene_gt_info_camX.json."""
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
                continue  # Skip objects that don't match the specified obj_id

            rotation_matrix = np.array(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            translation = np.array(obj["cam_t_m2c"], dtype=np.float32)  # [X, Y, Z]
            gt_poses.append(calc_pose_matrix(rotation_matrix, translation))

    return gt_poses

def calc_pose_matrix(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
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
