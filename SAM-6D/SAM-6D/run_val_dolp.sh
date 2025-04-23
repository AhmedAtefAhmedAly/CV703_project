#!/usr/bin/env bash
# stop on undefined vars, propagate pipe errors, but *don’t* auto‑exit on non‑zero exit codes
set -uo pipefail

##############################################################################
VAL_DIR="/l/users/ahmed.aly/ipd/val"
MODEL_ROOT="/l/users/ahmed.aly/ipd/models_eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEGMENTOR_MODEL="sam"
CAM_IDS=(cam1 cam2 cam3)
OBJ_IDS=(14)
##############################################################################

for OBJ_ID in "${OBJ_IDS[@]}"; do
  # zero‑pad to 6 digits (e.g. 000014)
  PADDED=$(printf "%06d" "${OBJ_ID}")
  CAD_MODEL="${MODEL_ROOT}/obj_${PADDED}.ply"
  OUTPUT_ROOT="/home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D/SAM-6D/Render/Data/IPD/OBJ_${OBJ_ID}_all_dolppt"
  GLOBAL_TEMPLATE_DIR="/home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D/sam6d_templates_dolppt_obj${OBJ_ID}"

  echo "▶▶▶ Processing OBJ ${OBJ_ID} (model: ${CAD_MODEL})"
  mkdir -p "$GLOBAL_TEMPLATE_DIR"

  # ────────── render templates once ──────────────────────────────────────────
  if ! compgen -G "$GLOBAL_TEMPLATE_DIR/templates/rgb_*.png" >/dev/null; then
    echo "    Rendering templates once → $GLOBAL_TEMPLATE_DIR"
    pushd "$SCRIPT_DIR/Render" >/dev/null
      blenderproc run render_custom_templates.py \
        --output_dir "$GLOBAL_TEMPLATE_DIR" \
        --cad_path   "$CAD_MODEL" 
    popd >/dev/null
  fi

  # ────────── main loops ─────────────────────────────────────────────────────
  for scene_path in "$VAL_DIR"/*/ ; do
    scene=$(basename "$scene_path")
    echo "  ⏳  Scene $scene (OBJ ${OBJ_ID})"

    for cam in "${CAM_IDS[@]}"; do
      rgb_dir="$scene_path/rgb_${cam}"
      depth_dir="$scene_path/depth_${cam}"
      dolp_dir="$scene_path/dolp_${cam}"
      cam_json="$scene_path/scene_camera_${cam}.json"
      gt_json="$scene_path/scene_gt_${cam}.json"

      # skip if any of rgb/depth/dolp folders are missing
      if [[ ! -d "$rgb_dir" || ! -d "$depth_dir" || ! -d "$dolp_dir" ]]; then
        echo "    ── $cam : missing rgb/depth/dolp → skip"
        continue
      fi
      echo "    ▶ Camera $cam"

      shopt -s nullglob
      for rgb_file in "$rgb_dir"/*.png; do
        file=$(basename "$rgb_file")
        depth_file="$depth_dir/$file"
        dolp_file="$dolp_dir/$file"
        image_id="${file%.*}"

        # check depth & DoLP exist
        [[ ! -f "$depth_file" ]] && echo "      depth $file missing → skip" && continue
        [[ ! -f "$dolp_file" ]]  && echo "      dolp  $file missing → skip" && continue

        OUT_DIR="$OUTPUT_ROOT/$scene/$cam/$image_id"
        mkdir -p "$OUT_DIR/templates"

        # copy template files if not already present
        if [[ ! -e "$OUT_DIR/templates/rgb_0.png" ]]; then
          cp -r "$GLOBAL_TEMPLATE_DIR/templates/"* "$OUT_DIR/templates/"
        fi

        # ---------------- instance segmentation -----------------------------
        echo "      • segmentation"
        pushd "$SCRIPT_DIR/Instance_Segmentation_Model" >/dev/null
        if ! python run_inference_dolp.py \
                --segmentor_model "$SEGMENTOR_MODEL" \
                --output_dir      "$OUT_DIR" \
                --cad_path        "$CAD_MODEL" \
                --rgb_path        "$rgb_file" \
                --depth_path      "$depth_file" \
                --cam_path        "$cam_json" \
                --dolp_path       "$dolp_file" ; then
          popd >/dev/null
          echo "        ⚠ segmentation failed – skip image"
          rm -rf "$OUT_DIR/templates"
          continue
        fi
        popd >/dev/null

        SEG_PATH="$OUT_DIR/sam6d_results/detection_ism.json"
        if [[ ! -s "$SEG_PATH" ]]; then
          echo "        ⚠ no detections – skip image"
          rm -rf "$OUT_DIR/templates"
          continue
        fi

        # -------------------- pose estimation -------------------------------
        echo "      • pose"
        pushd "$SCRIPT_DIR/Pose_Estimation_Model" >/dev/null
        if ! python run_inference_custom.py \
                --output_dir "$OUT_DIR" \
                --cad_path   "$CAD_MODEL" \
                --rgb_path   "$rgb_file" \
                --depth_path "$depth_file" \
                --cam_path   "$cam_json" \
                --seg_path   "$SEG_PATH"   \
                --gt_path    "$gt_json" ; then
          popd >/dev/null
          echo "        ⚠ pose failed – skip image"
          rm -rf "$OUT_DIR/templates"
          continue
        fi
        popd >/dev/null

        # success → clean up templates
        rm -rf "$OUT_DIR/templates"
      done
      shopt -u nullglob
    done
  done

  echo "✅  OBJ ${OBJ_ID} done. Results under: $OUTPUT_ROOT"
done

echo "🎉 All OBJ processed."
