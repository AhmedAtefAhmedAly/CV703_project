import numpy as np
from scipy.spatial.transform import Rotation as R
 
def compute_add(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    Compute Average Distance of Model Points (ADD).
    """
    model_gt = (R_gt @ model_points.T) + t_gt
    model_pred = (R_pred @ model_points.T) + t_pred
    dists = np.linalg.norm(model_gt - model_pred, axis=0)
    return np.mean(dists) * 1000  # convert to mm
 
 
def compute_mvd(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    Compute Mean Vertex Distance (MVD).
    """
    model_gt = (R_gt @ model_points.T) + t_gt
    model_pred = (R_pred @ model_points.T) + t_pred
    mvd = np.mean(np.linalg.norm(model_gt.T - model_pred.T, axis=1))
    return mvd * 1000  # convert to mm
 
 
def compute_rotation_translation_error(R_gt, t_gt, R_pred, t_pred):
    """
    Compute rotational and translational error between GT and predicted poses.
    Rotation in degrees, translation in mm.
    """
    # Rotation Error (in degrees)
    R_err = R.from_matrix(R_gt.T @ R_pred)
    angle_error = np.rad2deg(R_err.magnitude())
 
    # Translation Error (in mm)
    t_error = np.linalg.norm(t_gt - t_pred) * 1000  # convert to mm
 
    return angle_error, t_error
 