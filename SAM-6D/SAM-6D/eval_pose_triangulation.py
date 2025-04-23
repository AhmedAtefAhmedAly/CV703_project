#!/usr/bin/env python3
"""
eval_pose.py — Greedy 6‑D pose evaluation across GT and SAM‑6D predictions.

Usage example:
  python eval_pose.py \
    --gt_root   /l/users/ahmed.aly/ipd/val \
    --pred_root /home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D/SAM-6D/Render/Data/IPD \
    --cad_dir   /l/users/ahmed.aly/ipd/models \
    --objects   0 14 \
    --trans_thresh 100
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _rot_err_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos = np.clip((np.trace(R_pred @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def _pair_cost(
    R_pred: np.ndarray, t_pred: np.ndarray,
    R_gt: np.ndarray, t_gt: np.ndarray,
    lam: float = 1e-3
) -> Tuple[float, float, float]:
    r_err = _rot_err_deg(R_pred, R_gt)
    t_err = float(np.linalg.norm(t_pred - t_gt))
    return r_err + lam * t_err, r_err, t_err


def load_model_points(ply_path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ModuleNotFoundError:
        raise RuntimeError(
            "Binary PLY detected – please `pip install plyfile`."
        )
    ply = PlyData.read(str(ply_path))
    v   = ply["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return xyz


def evaluate_detections(
    preds: List[dict],
    gt_index: Dict[Tuple[int,int,int], dict],
    model_pts: np.ndarray,
    obj_id: int,
    image_id: int,
    score_thresh: float = 0.0,
    translation_error: float | None = 100.0,
) -> dict:
    # filter GT for this image & object
    relevant_gt = {
        k: v for k, v in gt_index.items()
        if k[2] == obj_id and k[0] == image_id
    }
    total_gt = len(relevant_gt)
    if total_gt == 0:
        empty = {"mean": None, "median": None, "all": []}
        return {
            "counts": {"total_gt":0,"matched":0,"missed":0,"extras":0},
            "recall": None, "precision": None,
            "rotation_deg": empty, "translation": empty, "MSSD": empty
        }

    free_gt = set(relevant_gt.keys())
    valid_preds = [
        p for p in preds
        if p["category_id"] == obj_id 
    ]
    # valid_preds.sort(key=lambda x: -x["score"])

    rot_errs, trans_errs, mssd_vals = [], [], []

    for p in valid_preds:
        if not free_gt:
            break

        R_p, t_p = np.asarray(p["R"]), np.asarray(p["t"])
        best_key, best_cost = None, float("inf")
        best_r, best_t = None, None

        for k in free_gt:
            R_g, t_g = relevant_gt[k]["R"], relevant_gt[k]["t"]
            cost, r_e, t_e = _pair_cost(R_p, t_p, R_g, t_g)
            if cost < best_cost:
                best_key, best_cost = k, cost
                best_r, best_t = r_e, t_e

        if translation_error is not None and best_t > translation_error:
            continue

        free_gt.remove(best_key)
        rot_errs.append(best_r)
        trans_errs.append(best_t)

        pts_pred = (R_p @ model_pts.T).T + t_p
        R_g, t_g = relevant_gt[best_key]["R"], relevant_gt[best_key]["t"]
        pts_gt   = (R_g @ model_pts.T).T + t_g
        mssd_vals.append(float(np.mean(np.sum((pts_pred - pts_gt)**2, axis=1))))

    matched = len(rot_errs)
    extras  = len(valid_preds) - matched
    missed  = len(free_gt)

    def _agg(v, fn): return float(fn(v)) if v else None
    def _stat(v): return {"mean": _agg(v, np.mean),
                          "median": _agg(v, np.median),
                          "all": v}

    return {
        "rotation_deg": _stat(rot_errs),
        "translation" : _stat(trans_errs),
        "MSSD"        : _stat(mssd_vals),
        "counts"      : {"total_gt": total_gt,
                         "matched": matched,
                         "missed": missed,
                         "extras": extras},
        "recall"    : matched / total_gt if total_gt else None,
        "precision" : matched / (matched + extras)
                       if (matched + extras) else None,
    }


def load_gt(scene_dir: Path, cam: int) -> Dict[Tuple[int,int,int], dict]:
    path = scene_dir / f"scene_gt_cam{cam}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: Dict[Tuple[int,int,int], dict] = {}
    for img_str, objs in raw.items():
        img_id = int(img_str)
        for inst_idx, o in enumerate(objs):
            R = np.asarray(o["cam_R_m2c"], np.float32).reshape(3,3)
            t = np.asarray(o["cam_t_m2c"], np.float32)
            out[(img_id, inst_idx, o["obj_id"])] = {"R":R,"t":t}
    return out

def load_cam_k(scene_dir: Path, cam: int) -> Dict[int, dict]:
    """
    Load camera intrinsics (K), extrinsics (R, t) and depth scale for
    camera index `cam` from the scene directory.

    Returns a dict mapping image_id -> {"K": np.ndarray(3,3),
                                        "R": np.ndarray(3,3),
                                        "t": np.ndarray(3,),
                                        "depth_scale": float}
    """
    cam_file = scene_dir / f"scene_camera_cam{cam}.json"
    if not cam_file.exists():
        # no camera file for this camera index
        return {}
    raw = json.loads(cam_file.read_text())
    out: Dict[int, dict] = {}
    for img_str, params in raw.items():
        try:
            img_id = int(img_str)
        except ValueError:
            # skip non-numeric keys
            continue
        # intrinsics
        K = np.asarray(params["cam_K"], dtype=np.float32).reshape(3, 3)
        # extrinsics: world-to-camera
        R = np.asarray(params.get("cam_R_w2c"), dtype=np.float32).reshape(3, 3)
        t = np.asarray(params.get("cam_t_w2c"), dtype=np.float32)
        depth_scale = float(params.get("depth_scale", 1.0))
        out[img_id] = {"K": K, "R": R, "t": t, "depth_scale": depth_scale}
    return out

def transform_pose_to_world(R_pred: np.ndarray,
                            t_pred: np.ndarray,
                            cam_params: dict) -> Tuple[np.ndarray,np.ndarray]:
    """
    Given a predicted camera‐frame pose (R_pred, t_pred) and camera params,
    return world‐frame pose (R_w, t_w).
    """
    R_w2c = cam_params["R"]
    t_w2c = cam_params["t"]
    # Rotation: from model to world
    R_world = R_w2c.T @ R_pred
    # Translation: invert x_cam = R_w2c @ x_w + t_w2c → x_w = R_w2c.T@(x_cam - t_w2c)
    t_world = R_w2c.T @ (t_pred - t_w2c)
    return R_world, t_world

def triangulate_world_point(proj_mats: List[np.ndarray],
                            img_pts: List[Tuple[float,float]]) -> np.ndarray:
    """
    Triangulate a single 3D point from multiple views.
    proj_mats: list of 3×4 camera projection matrices P = K @ [R|t]
    img_pts:   list of (u,v) pixel coordinates in each view
    Returns 3D point in world coords.
    """
    # Build linear system A X = 0
    A = []
    for P, (u, v) in zip(proj_mats, img_pts):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)  # shape (2*n, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float32)

def filter_multiview_consistency(
        detections: Dict[int, List[dict]],
        cam_params: Dict[int, Dict[int, dict]],
        img_id: int,
        model_pts: np.ndarray,
        pos_thresh: float = 50.0
    ) -> Dict[int, List[dict]]:
    """
    Keep only detections that have at least one other view
    within pos_thresh (mm) in world coordinates for the same img_id.
    Returns filtered detections keyed by cam_idx.
    """
    entries = []  # List[dict(cam_idx, pred, t_world)]
    for cam_idx, preds in detections.items():
        params_map = cam_params.get(cam_idx, {})
        params = params_map.get(img_id)
        if params is None:
            continue
        for p in preds:
            R_pred = np.asarray(p["R"], np.float32)
            t_pred = np.asarray(p["t"], np.float32)
            _, t_world = transform_pose_to_world(R_pred, t_pred, params)
            entries.append({"cam_idx": cam_idx, "pred": p, "t_world": t_world})

    inliers = []
    for e in entries:
        for o in entries:
            if o["cam_idx"] != e["cam_idx"]:
                if np.linalg.norm(e["t_world"] - o["t_world"]) < pos_thresh:
                    inliers.append((e["cam_idx"], e["pred"]))
                    break

    out: Dict[int, List[dict]] = defaultdict(list)
    for cam_idx, pred in inliers:
        out[cam_idx].append(pred)
    return out

def load_predictions(scene_dir: Path, cam: int) -> Dict[int, List[dict]]:
    d: Dict[int,List[dict]] = defaultdict(list)
    cam_dir = scene_dir / f"cam{cam}"
    if not cam_dir.exists():
        return d
    for img_dir in cam_dir.iterdir():
        if not img_dir.is_dir():
            continue
        pfile = img_dir / "sam6d_results" / "detection_pem.json"
        if not pfile.exists():
            continue
        try:
            preds = json.loads(pfile.read_text())
        except json.JSONDecodeError:
            continue
        d[int(img_dir.name)].extend(preds)
    return d


def evaluate_dataset(
    gt_root: Path,
    pred_root: Path,
    cad_dir: Path,
    obj_ids: List[int],
    pred_dir_pattern: str,
    cams: Tuple[int, ...] = (1, 2, 3),
    score_thresh: float = 0.0,
    trans_thresh: float = 100.0,
    multi_view_thresh: float = 10.0,
) -> dict:
    # 1) Load CAD model points
    model_pts = {
        oid: load_model_points(cad_dir / f"obj_{oid:06d}.ply")
        for oid in obj_ids
    }

    # 2) Stats accumulator
    stats = {
        oid: {"rot": [], "trans": [], "mssd": [],
              "TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for oid in obj_ids
    }

    # 3) Iterate scenes
    scenes = sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    for scene_id in scenes:
        scene_dir = gt_root / scene_id

        # pre‐load all cams' GT and cam‑files
        gt_indexes = {c: load_gt(scene_dir, c)   for c in cams}
        cam_params  = {c: load_cam_k(scene_dir, c) for c in cams}

        for oid in obj_ids:
            st = stats[oid]
            subdir = pred_dir_pattern.format(oid=oid)
            pred_obj_root = pred_root / subdir / scene_id

            # load every cam’s predictions
            preds_per_cam = {
                c: load_predictions(pred_obj_root, c)
                for c in cams
            }

            # gather all img_ids where this obj shows up
            img_ids = set()
            for c in cams:
                img_ids |= {k[0] for k in gt_indexes[c].keys() if k[2] == oid}
                img_ids |= set(preds_per_cam[c].keys())

            for img_id in img_ids:
                # build raw per‑cam lists
                dets = {
                    c: [p for p in preds_per_cam[c].get(img_id, [])
                        if p["category_id"] == oid]
                    for c in cams
                }

                filtered = filter_multiview_consistency(
                    dets,
                    cam_params,
                    img_id,                 # <— important!
                    model_pts[oid],
                    pos_thresh=multi_view_thresh
                )

                # now run single‑view eval on survivors
                for c in cams:
                    preds    = filtered.get(c, [])
                    gt_index = gt_indexes[c]
                    gt_sub   = {
                        k: v for k, v in gt_index.items()
                        if k[0] == img_id and k[2] == oid
                    }
                    n_gt = len(gt_sub)

                    if n_gt == 0 and not preds:
                        st["TN"] += 1; continue
                    if n_gt == 0 and preds:
                        st["FP"] += len(preds); continue
                    if n_gt > 0 and not preds:
                        st["FN"] += n_gt; continue

                    res = evaluate_detections(
                        preds, gt_index,
                        model_pts[oid],
                        oid, img_id,
                        score_thresh, trans_thresh
                    )
                    cts = res["counts"]
                    st["TP"] += cts["matched"]
                    st["FP"] += cts["extras"]
                    st["FN"] += cts["missed"]
                    st["rot"].extend(res["rotation_deg"]["all"])
                    st["trans"].extend(res["translation"]["all"])
                    st["mssd"].extend(res["MSSD"]["all"])

    # 4) Final aggregation
    def _agg(arr, fn): return float(fn(arr)) if arr else None

    summary = {}
    for oid, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        summary[oid] = {
            "precision":        tp / (tp + fp) if (tp + fp) else None,
            "recall":           tp / (tp + fn) if (tp + fn) else None,
            "translation_mean": _agg(s["trans"], np.mean),
            "rotation_mean":    _agg(s["rot"],   np.mean),
            "mssd_mean":        _agg(s["mssd"],  np.mean),
            **s
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_root",   required=True, type=Path,
                   help="ground‑truth root (scene dirs)")
    p.add_argument("--pred_root", required=True, type=Path,
                   help="parent of all OBJ_{id}_all folders")
    p.add_argument("--cad_dir",   required=True, type=Path,
                   help="folder with obj_000014.ply, …")
    p.add_argument("--objects",   nargs="+", type=int, required=True)
    p.add_argument("--pred_dir_pattern", type=str, default="outputs_{oid}",
                   help="format string for prediction subdirectories under --pred_root; must include “{oid}” (e.g. OBJ_{oid}_all_dolpp or outputs_{oid})")
    p.add_argument("--score_thresh", type=float, default=0.0)
    p.add_argument("--trans_thresh", type=float, default=100.0)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    res = evaluate_dataset(
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        cad_dir=args.cad_dir,
        obj_ids=args.objects,
        score_thresh=args.score_thresh,
        trans_thresh=args.trans_thresh,
        pred_dir_pattern = args.pred_dir_pattern
    )

    # print table with MSSD
    print("\nobj  recall  prec   t‑mean   rot‑mean°   mssd‑mean")
    print("----------------------------------------------------")
    for oid, r in res.items():
        print(f"{oid:>3}  "
              f"{(r['recall']    or 0):7.2f}  "
              f"{(r['precision'] or 0):6.2f}  "
              f"{(r['translation_mean'] or 0):7.1f}   "
              f"{(r['rotation_mean']    or 0):8.2f}   "
              f"{(r['mssd_mean']        or 0):9.1f}")

    # ─── overall metrics ───
    tp = sum(r["TP"] for r in res.values())
    fp = sum(r["FP"] for r in res.values())
    fn = sum(r["FN"] for r in res.values())

    overall_prec = tp / (tp + fp) if (tp + fp) else 0.0
    overall_rec  = tp / (tp + fn) if (tp + fn) else 0.0

    # collect all per‐match errors
    all_trans = [e for r in res.values() for e in r["trans"]]
    all_rot   = [e for r in res.values() for e in r["rot"]]
    all_mssd  = [e for r in res.values() for e in r["mssd"]]

    overall_t_mean    = float(np.mean(all_trans)) if all_trans else 0.0
    overall_r_mean    = float(np.mean(all_rot))   if all_rot   else 0.0
    overall_mssd_mean = float(np.mean(all_mssd))  if all_mssd  else 0.0

    print("\nOverall metrics")
    print("----------------")
    print(f"Precision:         {overall_prec:.2f}")
    print(f"Recall:            {overall_rec:.2f}")
    print(f"Trans. error mean:  {overall_t_mean:.1f}")
    print(f"Rot.   error mean:  {overall_r_mean:.2f}°")
    print(f"MSSD mean:          {overall_mssd_mean:.3f}")
    # ────────────────────────────

    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nSaved full JSON → {args.out}")


if __name__ == "__main__":
    main()
