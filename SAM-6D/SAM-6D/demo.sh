# Render CAD templates

export CAD_PATH=/l/users/ahmed.aly/ipd/models/obj_000014.ply   # path to a given cad model(mm)
export RGB_PATH=/l/users/ahmed.aly/ipd/val/000008/rgb_cam3/000005.png         # path to a given RGB image
export DEPTH_PATH=/l/users/ahmed.aly/ipd/val/000008/depth_cam3/000005.png       # path to a given depth map(mm)
export CAMERA_PATH=/l/users/ahmed.aly/ipd/val/000008/scene_camera_cam3.json  # path to given camera intrinsics   # path to given camera intrinsics
export DOLP_PATH=/l/users/ahmed.aly/ipd/val/000008/dolp_cam3/000005.png   # path to a given DOLP image
export OUTPUT_DIR=/home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D/SAM-6D/Render/Data/IPD/OBJ_14        # path to a pre-defined file for saving results
export GT_PATH=/l/users/ahmed.aly/ipd/val/000008/scene_gt_cam3.json

cd Render &&
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH && #--colorize &&

# Run instance segmentation model
export SEGMENTOR_MODEL=sam
cd ../Instance_Segmentation_Model &&
python run_inference_dolp.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --dolp_path $DOLP_PATH &&


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json
cd ../Pose_Estimation_Model &&
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH --gt_path $GT_PATH 

