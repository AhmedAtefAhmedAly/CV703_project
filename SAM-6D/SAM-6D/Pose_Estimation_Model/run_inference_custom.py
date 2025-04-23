import gorilla
import argparse
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json

import torch
import torchvision.transforms as transforms
import cv2

# Add your project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path += [
    osp.join(ROOT_DIR, 'provider'),
    osp.join(ROOT_DIR, 'utils'),
    osp.join(ROOT_DIR, 'model'),
    osp.join(BASE_DIR, 'model', 'pointnet2'),
]

# -----------------------------------------------------------------------------
# Argument parsing & config
# -----------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(description="Pose Estimation with Evaluation")
    parser.add_argument("--gpus",            type=str,   default="0", help="GPU ids")
    parser.add_argument("--model",           type=str,   default="pose_estimation_model", help="Model file")
    parser.add_argument("--config",          type=str,   default="config/base.yaml", help="Config file")
    parser.add_argument("--iter",            type=int,   default=600000, help="Epoch num for testing")
    parser.add_argument("--exp_id",          type=int,   default=0, help="Experiment ID")
    parser.add_argument("--output_dir",      required=True, help="Root output directory")
    parser.add_argument("--cad_path",        required=True, help="Path to CAD model (mm)")
    parser.add_argument("--rgb_path",        required=True, help="Path to RGB image")
    parser.add_argument("--depth_path",      required=True, help="Path to depth image (mm)")
    parser.add_argument("--cam_path",        required=True, help="Path to camera info JSON")
    parser.add_argument("--seg_path",        required=True, help="Path to segmentation JSON")
    parser.add_argument("--gt_path",         required=True, help="Path to ground-truth JSON")
    parser.add_argument("--det_score_thresh", type=float, default=0.2, help="Detection score threshold")
    return parser.parse_args()

def init():
    args = get_parser()
    exp_name = f"{args.model}_{osp.splitext(args.config)[0]}_id{args.exp_id}"
    log_dir  = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name         = exp_name
    cfg.gpus             = args.gpus
    cfg.model_name       = args.model
    cfg.log_dir          = log_dir
    cfg.test_iter        = args.iter
    cfg.output_dir       = args.output_dir
    cfg.cad_path         = args.cad_path
    cfg.rgb_path         = args.rgb_path
    cfg.depth_path       = args.depth_path
    cfg.cam_path         = args.cam_path
    cfg.seg_path         = args.seg_path
    cfg.gt_path          = args.gt_path
    cfg.det_score_thresh = args.det_score_thresh

    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return cfg

# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
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

def load_ipd_gt(gt_path):
    raw = json.load(open(gt_path))
    gt_index = {}
    for scene_str, entries in raw.items():
        scene_id = int(scene_str)
        for img_id, entry in enumerate(entries):
            key = (scene_id, img_id, entry['obj_id'])
            R = np.array(entry['cam_R_m2c']).reshape(3,3)
            t = np.array(entry['cam_t_m2c'])
            gt_index[key] = {'R': R, 't': t}
    return gt_index

import numpy as np

def _rot_err_deg(R_pred, R_gt):
    """Geodesic distance (deg) between two rotation matrices."""
    cos = np.clip((np.trace(R_pred @ R_gt.T) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def _pair_cost(R_pred, t_pred, R_gt, t_gt, lam=1e-3):
    """Cost = rotation‑error + lam * translation‑error."""
    r_err = _rot_err_deg(R_pred, R_gt)
    t_err = np.linalg.norm(t_pred - t_gt)          # mm
    return r_err + lam * t_err, r_err, t_err       # return all three

# --------------------------------------------------------------------------- #
#  drop‑in replacement
# --------------------------------------------------------------------------- #
import numpy as np

def evaluate_detections(preds, gt_index, model_points,
                        obj_id, image_id,
                        score_thresh=0.0,
                        translation_error=100):
    """
    Greedy matching with an optional *translation‑error* threshold.

    A prediction is counted as a true match **only** if the translation error
    of the best assignment is ≤ ``trans_thresh``.  
    Otherwise the prediction is labelled an *extra* (false‑positive) and the
    ground‑truth instance stays in the pool for later predictions.

    Parameters
    ----------
    preds : list[dict]
    gt_index : dict[(img_id, inst_id, obj_id) → dict(R, t)]
    model_points : (N, 3) ndarray
    obj_id : int
    image_id : int
    score_thresh : float, optional
    trans_thresh : float | None, optional
        Maximum admissible translation error (same unit as *t*) for a match.
        If ``None`` → threshold disabled (original behaviour).

    Returns
    -------
    dict               # JSON‑serialisable
        ├─ rotation_deg : {mean, median, all}
        ├─ translation  : {mean, median, all}
        ├─ MSSD         : {mean, median, all}
        ├─ counts       : {total_gt, matched, missed, extras}
        ├─ recall
        └─ precision
    """

    # ------------------------------------------------------------------ GT --
    relevant_gt = {k: v for k, v in gt_index.items()
                   if k[2] == obj_id and k[0] == image_id}
    total_gt = len(relevant_gt)
    if total_gt == 0:
        empty = {'mean': None, 'median': None, 'all': []}
        return {'counts': {'total_gt': 0, 'matched': 0,
                           'missed': 0, 'extras': 0},
                'recall': None, 'precision': None,
                'rotation_deg': empty,
                'translation': empty,
                'MSSD': empty}

    free_gt = set(relevant_gt.keys())

    # ------------------------------------------------------------ predictions
    valid_preds = [p for p in preds
                   if p['category_id'] == obj_id and p['score'] >= score_thresh]
    valid_preds.sort(key=lambda x: -x['score'])          # high → low

    rot_errs, translation_errs, mssd_vals = [], [], []

    for p in valid_preds:
        if not free_gt:                                  # nothing left to match
            break

        # ---------- find best GT among the yet‑unmatched ones --------------
        best_key, best_cost = None, float('inf')
        best_rot, best_error = None, None
        R_p, t_p = np.asarray(p['R']), np.asarray(p['t'])

        for k in free_gt:
            R_g, t_g = relevant_gt[k]['R'], relevant_gt[k]['t']
            _, r_e, t_e = _pair_cost(R_p, t_p, R_g, t_g)
            cost = t_e
            if cost < best_cost:
                print(f"cost: {cost}, r_e: {r_e}, t_e: {t_e}")
                best_key, best_cost = k, cost
                best_rot, best_error = r_e, t_e

        # --------- accept or reject depending on translation error ---------
        if translation_error is not None and best_error > translation_error:
            # translation error too large → prediction stays *extra*
            continue

        # ------------------------- accept match ----------------------------
        free_gt.remove(best_key)                         
        rot_errs.append(best_rot)
        translation_errs.append(best_error)

        pts_pred = (R_p @ model_points.T).T + t_p
        R_g, t_g = relevant_gt[best_key]['R'], relevant_gt[best_key]['t']
        pts_gt   = (R_g @ model_points.T).T + t_g
        mssd_vals.append(float(np.mean(np.sum((pts_pred - pts_gt) ** 2, axis=1))))

    # ---------------------------------------------------------------- stats
    matched = len(rot_errs)            # #successful matches
    extras  = len(valid_preds) - matched
    missed  = len(free_gt)

    def _agg(lst, fn):
        return float(fn(lst)) if lst else None

    def _stat(lst):
        return {'mean': _agg(lst, np.mean),
                'median': _agg(lst, np.median),
                'all': lst}            # keep every error for later analysis

    return {
        'rotation_deg': _stat(rot_errs),
        'translation' : _stat(translation_errs),
        'MSSD'        : _stat(mssd_vals),
        'counts'      : {'total_gt': total_gt,
                         'matched': matched,
                         'missed': missed,
                         'extras': extras},
        'recall'    : matched / total_gt if total_gt else None,
        'precision' : matched / (matched + extras) if (matched + extras) else None
    }




# -----------------------------------------------------------------------------
# Data & visualization utils
# -----------------------------------------------------------------------------
from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections
import pycocotools.mask as cocomask
import trimesh

rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz

def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose

def parse_cam_id_from_path(cam_path):
    # e.g. ".../val/000008/scene_camera_cam1.json"
    folder = osp.basename(osp.dirname(cam_path))  # "scene_camera_cam1"
    camid  = ''.join(filter(str.isdigit, folder)) or '0'
    return camid

def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path,
                  det_score_thresh, cfg):
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
    for det in dets_:
        if det['score'] > det_score_thresh:
            dets.append(det)
    del dets_

    # camera intrinsics
    cam_info = json.load(open(cam_path))
    cam_id   = parse_cam_id_from_path(cam_path)
    if cam_id not in cam_info:
        cam_id = next(iter(cam_info.keys()))
    K = np.array(cam_info[cam_id]['cam_K']).reshape(3,3)

    # full image & point cloud
    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info[cam_id]['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    # CAD model pts for eval later
    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))


    # prepare each instance
    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']
        
        # mask

        size = seg["size"]

        if len(size) == 2:
            h, w = size
        elif len(size) == 3 and size[0] == 1:
            _, h, w = size           # drop the dummy first dimension
        else:
            raise ValueError(f"Unexpected RLE size format: {size}")
        
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg

                    
        # accept [1, H, W] as well as [H, W]
        size = rle["size"]
        if len(size) == 3 and size[0] == 1:
            rle = {"size": size[1:], "counts": rle["counts"]}

        mask = cocomask.decode(rle)  # (H, W) uint8 mask
        
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]
        
        # subsample
        N_obs = cfg.n_sample_observed_point
        if len(choose)<=N_obs:
            idxs = np.random.choice(len(choose), N_obs)
        else:
            idxs = np.random.choice(len(choose), N_obs, replace=False)
        choose, cloud = choose[idxs], cloud[idxs]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = init()
    random.seed(cfg.rd_seed); torch.manual_seed(cfg.rd_seed)

    scene_id, cam_id, image_id = parse_ids_from_paths(
        cfg.rgb_path, cfg.depth_path, cfg.cam_path
    )

    obj_id = parse_obj_id_from_cad(cfg.cad_path)

    # model setup
    print("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model).cuda().eval()
    ckpt  = osp.join(osp.dirname(__file__), 'checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=model, filename=ckpt)

    # templates
    print("=> extracting templates ...")
    tem_p = osp.join(cfg.output_dir, 'templates')
    all_tem, all_tem_pts, all_tem_ch = get_templates(tem_p, cfg.test_dataset)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(
            all_tem, all_tem_pts, all_tem_ch
        )

    # data prep
    print("=> loading input data ...")
    batch, img, whole_pts, model_points, detections = get_test_data(
        cfg.rgb_path, cfg.depth_path, cfg.cam_path,
        cfg.cad_path, cfg.seg_path,
        cfg.det_score_thresh, cfg.test_dataset
    )
    ninst = batch['pts'].size(0)

    # inference
    print("=> running model ...")
    with torch.no_grad():
        batch['dense_po'] = all_tem_pts.repeat(ninst,1,1)
        batch['dense_fo'] = all_tem_feat.repeat(ninst,1,1)
        out = model(batch)

    # scores & poses
    pose_scores = (out.get('pred_pose_score', out['score']) * out['score']) \
                  .cpu().numpy()
    pred_rot     = out['pred_R'].cpu().numpy()
    pred_trans  = out['pred_t'].cpu().numpy() * 1000

    # save detections
    print("=> saving results ...")
    res_dir = osp.join(cfg.output_dir, 'sam6d_results')
    os.makedirs(res_dir, exist_ok=True)
    for i, det in enumerate(detections):
        det['score'] = float(pose_scores[i])
        det['R']     = pred_rot[i].tolist()
        det['t']     = pred_trans[i].tolist()
    det_path = osp.join(res_dir, 'detection_pem.json')
    with open(det_path, 'w') as f:
        json.dump(detections, f)

    # evaluation
    print("=> evaluating ...")
    gt_index = load_ipd_gt(cfg.gt_path)
    metrics  = evaluate_detections(detections, gt_index, model_points, obj_id=obj_id, image_id=image_id, score_thresh=0.0)
    eval_path = osp.join(res_dir, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # print summary
    print("=== EVALUATION ===")
    print(json.dumps(metrics, indent=2))

    # visualization
    print("=> visualizing ...")
    save_v = osp.join(res_dir, 'vis_pem.png')
    mask   = pose_scores > -1  # all
    K_vis  = batch['K'][mask].cpu().numpy()
    vis_img = visualize(img,
                        pred_rot[mask],
                        pred_trans[mask],
                        (model_points*1000),
                        K_vis,
                        save_v)
    vis_img.save(save_v)

