#!/usr/bin/env bash
# stop on undefined vars, propagate pipe errors, but *don’t* auto‑exit on non‑zero
set -uo pipefail

##############################################################################
VAL_DIR="/l/users/ahmed.aly/ipd/val"
MODEL_ROOT="/l/users/ahmed.aly/ipd/models_eval"
OUTPUT_BASE="/home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D/SAM-6D/Render/Data/IPD" ###directory of your output folder could be downloaded from github
TEMPLATE_BASE="/home/ahmed.aly/Projects/Bin-Picking/bpc_baseline/SAM-6D"
SEGMENTOR_MODEL="sam"
CAM_IDS=(cam1 cam2 cam3)
OBJ_IDS=(0 1 14)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for OBJ_ID in "${OBJ_IDS[@]}"; do
  # zero‑pad to 6 digits (e.g. 000014)
  PADDED=$(printf "%06d" "${OBJ_ID}")
  CAD_MODEL="${MODEL_ROOT}/obj_${PADDED}.ply"
  OUTPUT_ROOT="${OUTPUT_BASE}/OBJ_${OBJ_ID}_alll"
  GLOBAL_TEMPLATE_DIR="${TEMPLATE_BASE}/sam6d_templates_dolp_obj${OBJ_ID}"

  echo "▶▶▶ Processing OBJ ${OBJ_ID} (CAD = ${CAD_MODEL})"
  mkdir -p "$GLOBAL_TEMPLATE_DIR"

  # ────────── render templates once ──────────────────────────────────────────
  if ! compgen -G "$GLOBAL_TEMPLATE_DIR/templates/rgb_*.png" >/dev/null; then
    echo "    Rendering templates → $GLOBAL_TEMPLATE_DIR"
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
      cam_json="$scene_path/scene_camera_${cam}.json"
      gt_json="$scene_path/scene_gt_${cam}.json"

      [[ -d "$rgb_dir" && -d "$depth_dir" ]] || {
        echo "    ── $cam : rgb/depth missing → skip"
        continue
      }
      echo "    ▶ Camera $cam"

      shopt -s nullglob
      for rgb_file in "$rgb_dir"/*.png; do
        file=$(basename "$rgb_file")
        depth_file="$depth_dir/$file"
        image_id="${file%.*}"

        [[ -f "$depth_file" ]] || {
          echo "      depth $file missing → skip"
          continue
        }

        OUT_DIR="$OUTPUT_ROOT/$scene/$cam/$image_id"
        mkdir -p "$OUT_DIR/templates"

        # copy template files if not present
        [[ -e "$OUT_DIR/templates/rgb_0.png" ]] || \
          cp -r "$GLOBAL_TEMPLATE_DIR/templates/"* "$OUT_DIR/templates/"

        # ---------------- instance segmentation -----------------------------
        echo "      • segmentation"
        pushd "$SCRIPT_DIR/Instance_Segmentation_Model" >/dev/null
        if ! python run_inference_custom.py \
                --segmentor_model "$SEGMENTOR_MODEL" \
                --output_dir      "$OUT_DIR" \
                --cad_path        "$CAD_MODEL" \
                --rgb_path        "$rgb_file" \
                --depth_path      "$depth_file" \
                --cam_path        "$cam_json" ; then
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
                --seg_path   "$SEG_PATH" \
                --gt_path    "$gt_json" ; then
          popd >/dev/null
          echo "        ⚠ pose failed – skip image"
          rm -rf "$OUT_DIR/templates"
          continue
        fi
        popd >/dev/null

        # success → clean up
        rm -rf "$OUT_DIR/templates"
      done
      shopt -u nullglob
    done
  done

  echo "✅  OBJ ${OBJ_ID} done. Results under: $OUTPUT_ROOT"
done

echo "🎉 All OBJ processed."
