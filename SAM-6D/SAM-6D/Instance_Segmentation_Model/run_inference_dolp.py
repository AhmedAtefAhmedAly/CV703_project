import os, sys
import numpy as np
import torch
import logging
import os.path as osp
import glob
import trimesh
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import time
from torchvision.utils import save_image
import torchvision.transforms as T
import imageio.v2 as imageio
import cv2 

from utils.poses.pose_utils import get_obj_poses_from_template_level
from model.utils import Detections, convert_npz_to_json
from utils.inout import load_json, save_json_bop23

# set level logging
logging.basicConfig(level=logging.INFO)

inv_rgb_transform = T.Compose([
    T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    ),
])

def visualize(rgb, detections, save_path="tmp.png"):
    import cv2, numpy as np
    from PIL import Image as PILImage
    import distinctipy
    from scipy.ndimage import binary_dilation
    from segment_anything.utils.amg import rle_to_mask

    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if len(detections) == 0:
        logging.warning("No detections found.")
        return rgb
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33
    for i, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = cv2.Canny((mask * 255).astype(np.uint8), 100, 200) > 0
        edge = binary_dilation(edge, np.ones((2, 2)))
        r_col, g_col, b_col = (int(255 * c) for c in colors[i])
        img[mask, 0] = alpha * r_col + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g_col + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b_col + (1 - alpha) * img[mask, 2]
        img[edge, :] = 255
    img_out = PILImage.fromarray(np.uint8(img))
    img_out.save(save_path)
    concat = PILImage.new('RGB', (rgb.width * 2, rgb.height))
    concat.paste(rgb, (0, 0))
    concat.paste(img_out, (rgb.width, 0))
    return concat


def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_calib = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    basename = os.path.splitext(os.path.basename(depth_path))[0]
    key = str(int(basename))
    if key not in cam_calib:
        logging.warning(f"Key {key} not in {cam_path}, using default key '0'.")
        key = "0"
    calib_entry = cam_calib[key]
    if 'cam_K' not in calib_entry or 'depth_scale' not in calib_entry:
        raise KeyError(f"Calibration for image {key} is missing 'cam_K' or 'depth_scale'")
    cam_K = np.array(calib_entry['cam_K']).reshape((3, 3))
    depth_scale = np.array(calib_entry['depth_scale'])
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch["depth_scale"] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def enhance_rgb_with_dolp(rgb, dolp, k=0.5, blur_size=5):
    """
    Enhance an RGB image by boosting its brightness based on the DoLP map.
 
    Parameters:
    - rgb:    uint8 RGB image, shape (H, W, 3), values in [0,255]
    - dolp:   float DoLP map, shape (H, W), normalized to [0,1]
    - k:      float, strength of the boost (e.g. 0.3–1.0)
    - blur_size: int, kernel size for smoothing DoLP to avoid noise artifacts.
 
    Returns:
    - enhanced_rgb: uint8 RGB image, same shape as input
    """
    # 1. Smooth DoLP to reduce noise
    dolp_smooth = cv2.GaussianBlur(dolp.astype(np.float32), (blur_size, blur_size), 0)
 
    # 2. Convert RGB → HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
 
    # 3. Boost V channel by DoLP
    #    hsv[...,2] is the V channel in [0,255]
    hsv[..., 2] += k * dolp_smooth * 255
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
 
    # 4. Back to RGB
    enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return enhanced_rgb


def load_gray(path):
    """
    Load any image as a true single‑channel grayscale.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        if np.allclose(r, g) and np.allclose(r, b):
            img = r
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def enhance_gray_with_dolp(gray, dolp,
                           alpha=1.0,
                           blur_size=7,
                           clahe_clip=4.0,
                           clahe_tiles=(4,4)):
    """
    Enhance a single‑channel grayscale image using DoLP to drive CLAHE + brightness.
    Returns a single‑channel uint8 result in [0–255].
    """
    d = dolp.astype(np.float32)
    if d.max() > 1.0:
        d /= 255.0
    d = cv2.GaussianBlur(d, (blur_size, blur_size), 0)
    d = np.clip(d, 0.0, 1.0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
    gray_eq = clahe.apply(gray)
    out = (1.0 - alpha * d) * gray + (alpha * d) * gray_eq
    return np.clip(out, 0, 255).astype(np.uint8)



def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh,scene_id,image_id,dolp_path,  cam_id, obj_id):
    
    # Hydra config
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError(f"The segmentor_model {segmentor_model} is not supported!")

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move descriptor and segmentor to device
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")

    # Prepare templates
    template_dir = osp.join(output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        img = Image.open(osp.join(template_dir, f'rgb_{idx}.png'))
        msk = Image.open(osp.join(template_dir, f'mask_{idx}.png'))
        boxes.append(msk.getbbox())
        t_img = torch.from_numpy(np.array(img.convert("RGB")) / 255.0).float()
        t_msk = torch.from_numpy(np.array(msk.convert("L")) / 255.0).float()
        templates.append(t_img * t_msk[:, :, None])
        masks.append(t_msk.unsqueeze(-1))
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)

    # Crop/resize pads
    from utils.bbox_utils import CropResizePad
    proc_cfg = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(proc_cfg.image_size)
    templates = proposal_processor(images=templates, boxes=torch.tensor(boxes)).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=torch.tensor(boxes)).to(device)

    # Compute descriptors for templates
    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
        templates, token_name="x_norm_clstoken"
    ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
        templates, masks_cropped[:, 0, :, :]
    ).unsqueeze(0).data

    # Load flattened template poses [N_total_poses, 4,4]
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses_flat = torch.tensor(template_poses).float().to(device)
    model.ref_data["poses"] = poses_flat  # now flat [N_total_poses, 4,4]

    # Sample CAD pointcloud
    mesh = trimesh.load_mesh(cad_path)
    pts = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(pts).unsqueeze(0).to(device)

    # Inference on one image
    gray = load_gray(rgb_path)
    dolp = cv2.imread(dolp_path, cv2.IMREAD_UNCHANGED)
    if dolp is None:
        raise FileNotFoundError(f"Could not load DoLP: {dolp_path}")
    dolp = dolp.astype(np.float32)
    if dolp.max() > 1.0:
        dolp /= 255.0

    enh = enhance_gray_with_dolp(gray, dolp)
    # gamma correction & stretch
    enh_f = (enh.astype(np.float32) / 255.0) ** 0.6
    enh = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)
    lo, hi = np.percentile(enh, (1, 99))
    enh = np.clip((enh - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    orig_rgb     = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    enhanced_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    # side_by_side = np.hstack((orig_rgb, enhanced_rgb))
    # dbg_dir = osp.join(output_dir, 'debug'); os.makedirs(dbg_dir, exist_ok=True)
    # Image.fromarray(side_by_side).save(osp.join(dbg_dir, f"{scene_id:06d}_{image_id:06d}_enh.png"))

    rgb = Image.fromarray(enhanced_rgb)    
    
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    print(len(detections))


############ Debugging
    # visualize(rgb, dets, osp.join(output_dir, 'debug', '01_raw_segmentation.png'))
    q_desc, q_appe_desc = model.descriptor_model.forward(np.array(rgb), detections)

    # Semantic matching: expected 4 returns
    idx_sel, pred_objs, sem_score, best_flat = model.compute_semantic_score(q_desc)
    detections.filter(idx_sel)
    print(len(detections))


############ Debugging
    # visualize(rgb, dets, osp.join(output_dir, 'debug', '02_semantic_filtered.png'))

    # Appearance matching
    q_appe_desc = q_appe_desc[idx_sel, :]
    appe_scores, ref_aux = model.compute_appearance_score(best_flat, pred_objs, q_appe_desc)


############ Debugging
    # topk = torch.topk(appe_scores, k=min(5, len(appe_scores)))[1]
    # appe_dets = detections.clone()
    # appe_dets.filter(topk)    
    # visualize(rgb, appe_dets, osp.join(output_dir, 'debug', '03_appearance_topk.png'))

    # Project chosen pose into image using flat index
    batch = batch_input_data(depth_path, cam_path, device)
    image_uv = model.project_template_to_image(best_flat, pred_objs, batch, detections.masks)

    # Geometric scoring
    geo_score, vis_ratio = model.compute_geometric_score(
        image_uv, detections, q_appe_desc, ref_aux, visible_thred=model.visible_thred
    )

############ Debugging
    # uv_img = np.array(rgb).copy()
    # for pts in image_uv:
    #     for u,v in pts.cpu().numpy().astype(int):
    #         cv2.circle(uv_img, (u,v), radius=2, color=(0,255,0), thickness=-1)
    # Image.fromarray(uv_img).save(osp.join(output_dir, 'debug', '04_projected_uv.png'))
############

    # Final score
    final_score = (sem_score + appe_scores + geo_score * vis_ratio) / (2 + vis_ratio)
    detections.add_attribute("scores", final_score)
    obj_ids = torch.full_like(final_score, obj_id, dtype=torch.long)

    detections.add_attribute("object_ids", obj_ids)
    # detections.add_attribute("object_ids", torch.zeros_like(final_score))

    # Save results
    detections.to_numpy()
    save_path = osp.join(output_dir, 'sam6d_results', 'detection_ism')
    detections.save_to_file(scene_id, image_id, 0, save_path, "Custom", return_results=False)
    dets = convert_npz_to_json(0, [save_path + ".npz"])
    save_json_bop23(save_path + ".json", dets)

    # Visualization
    vis = visualize(rgb, dets, osp.join(output_dir, 'sam6d_results', 'vis_ism.png'))
    vis.save(osp.join(output_dir, 'sam6d_results', 'vis_ism.png'))



def parse_ids_from_paths(rgb_path, depth_path, cam_path):
    # e.g. "/…/val/000008/rgb_cam1/000003.png"
    # split off image filename
    image_file = os.path.basename(rgb_path)           # "000003.png"
    image_id   = int(os.path.splitext(image_file)[0]) # 3

    cam_folder = os.path.basename(os.path.dirname(rgb_path))  # "rgb_cam1"
    cam_id     = int(cam_folder.split("cam")[-1])            # 1

    scene_folder = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))  # "000008"
    scene_id     = int(scene_folder)                                 # 8

    return scene_id, cam_id, image_id

def parse_obj_id_from_cad(cad_path):
    fname  = os.path.basename(cad_path)                   # "obj_000014.ply"
    stem   = os.path.splitext(fname)[0]                   # "obj_000014"
    obj_id = int(stem.split("_")[-1])                     # 14
    return obj_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cad_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--cam_path", required=True)
    parser.add_argument("--dolp_path", required=True)
    parser.add_argument("--stability_score_thresh", type=float, default=0.97)
    args = parser.parse_args()
    scene_id, cam_id, image_id = parse_ids_from_paths(
        args.rgb_path, args.depth_path, args.cam_path
    )

    obj_id = parse_obj_id_from_cad(args.cad_path)

    os.makedirs(osp.join(args.output_dir, 'sam6d_results'), exist_ok=True)
    run_inference(
        args.segmentor_model,
        args.output_dir,
        args.cad_path,
        args.rgb_path,
        args.depth_path,
        args.cam_path,
        dolp_path = args.dolp_path,
        stability_score_thresh=args.stability_score_thresh,
        scene_id= scene_id,
        cam_id= cam_id,
        image_id= image_id,
        obj_id= obj_id
    )