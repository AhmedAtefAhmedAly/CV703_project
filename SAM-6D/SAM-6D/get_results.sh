#!/bin/bash

##### User config #####
export GT_ROOT=/l/users/ahmed.aly/ipd/val                                       # Full path to the validation data
export PRED_ROOT=/home/ahmed.aly/Projects/CV703/SAM-6D/SAM-6D/Render/Data/IPD/ # Full path to the predicted data folder
export CAD_DIR=/l/users/ahmed.aly/ipd/models                                    # Full path to the CAD models folder
export OBJECTS=(0 1 14)                                                         # List of object IDs to evaluate


##### Results of baseline + depth #####
echo ""
echo "▶️  Evaluating baseline + depth results"
echo "--------------------------------------"

python eval_pose.py \
    --gt_root "$GT_ROOT" \
    --pred_root "$PRED_ROOT" \
    --cad_dir "$CAD_DIR" \
    --objects "${OBJECTS[@]}" \
    --trans_thresh 100 \
    --pred_dir_pattern outputs_{oid}

##### Results of SAM-6D #####
echo ""
echo "▶️  Evaluating SAM-6D results"
echo "-----------------------------"

python eval_pose.py \
    --gt_root "$GT_ROOT" \
    --pred_root "$PRED_ROOT" \
    --cad_dir "$CAD_DIR" \
    --objects "${OBJECTS[@]}" \
    --trans_thresh 100 \
    --pred_dir_pattern OBJ_{oid}_all


##### Results of SAM-6D #####
echo ""
echo "▶️  Evaluating SAM-6D results + post processing"
echo "-----------------------------"

python eval_pose_triangulation.py \
    --gt_root "$GT_ROOT" \
    --pred_root "$PRED_ROOT" \
    --cad_dir "$CAD_DIR" \
    --objects "${OBJECTS[@]}" \
    --trans_thresh 100 \
    --pred_dir_pattern OBJ_{oid}_all

##### Results of SAM-6D #####
echo ""
echo "▶️  Evaluating SAM-6D + Preprocessing results"
echo "-----------------------------"

python eval_pose.py \
    --gt_root "$GT_ROOT" \
    --pred_root "$PRED_ROOT" \
    --cad_dir "$CAD_DIR" \
    --objects "${OBJECTS[@]}" \
    --trans_thresh 100 \
    --pred_dir_pattern OBJ_{oid}_all_dolp


##### Results of SAM-6D #####
echo ""
echo "▶️  Evaluating SAM-6D + Preprocessing results + Postprocessing"
echo "-----------------------------"

python eval_pose_triangulation.py \
    --gt_root "$GT_ROOT" \
    --pred_root "$PRED_ROOT" \
    --cad_dir "$CAD_DIR" \
    --objects "${OBJECTS[@]}" \
    --trans_thresh 100 \
    --pred_dir_pattern OBJ_{oid}_all_dolp
