# Perception for Bin-picking

Multi-modal 6DoF pose estimation for industrial bin-picking using RGB, depth, and polarization with SAM-6D and SimplePoseNet

## Overview

This project tackles the challenge of accurate 6-DoF object pose estimation in cluttered industrial bin-picking scenarios using a multi-camera setup. It reproduces and extends two baseline methods—SimplePoseNet and SAM-6D—by incorporating:

- Depth fusion into SimplePoseNet for enhanced robustness.
- A DoLP-based image preprocessing pipeline to improve segmentation quality in SAM-6D.
- A multi-view spatial fusion step to improve pose accuracy and reduce false positives.

## Getting Started
### Requirements
This code was test on
- Python 3.12
- linux
- cuda toolkit 12

Create a conda environment with python 3.12
```
conda create -n your_environment python=3.12
pip install -r requirements.txt
```

### Dataset
This project uses the **IPD dataset** for training and evaluation.  
To download the dataset, follow the official instructions here:

👉 [IPD Dataset on Hugging Face](https://huggingface.co/datasets/bop-benchmark/ipd)

**Note**: If you do not have access to 7z, you can use cat and unzip the files for example 
```bash
cat ipd_train_pbr.z01 ipd_train_pbr.z02 ipd_train_pbr.z03 ipd_train_pbr.zip > ipd_train_pbr_full.zip
unzip ipd_train_pbr_full.zip -d ipd
```
### Download the model checkpoints
Download the models for yolo detection and PoseNet [here](https://drive.google.com/file/d/16Lch8q4R2-dBmo2SEiMzoKDJjQjL--A4/view?usp=sharing)

To download the models used in SAM-6D
```bash
cd SAM-6D/SAM-6D
bash prepare.sh
cd ../..
```

### Download the outputs
Download the final outputs produced by the models on object 0,1,14 [here](https://drive.google.com/file/d/1LO1NvslwVWEfiXk8wJ-8sSHgV4wZCcUt/view?usp=sharing)

## PoseNet
### Training
To train a detector model on a specific object. First, convert the data to YOLO format 

```
python PoseNet/bpc/yolo/prepare_data.py \
    --dataset_path "path/to/your/dataset/train_pbr" \
    --output_path "/path/to/save/the/data" \
    --obj_id 8
```
To train a baseline PoseNet model or depth model. The path to your dataset should be the root directory. 
```
python train_pose.py \ 
  --root_dir path/to/your/dataset \
  --target_obj_id 14 \
  --epochs 10 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_workers 16 \
  --checkpoints_dir bpc/pose/pose_checkpoints/ \
  --loss_type quat \
  --use_real_val
```
```
python train_pose_depth.py \
  --root_dir path/to/your/dataset \
  --target_obj_id 14 \
  --epochs 10 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_workers 16 \
  --checkpoints_dir bpc/pose/pose_checkpoints/depth/ \
  --loss_type quat \
  --use_real_val
```

### Evaluation
To create the output json files used for evaluation refer to the inference notebook. 

## SAM-6D
To run the evaluation of SAM-6D edit the correct paths to your dataset and prediction output folder in the bash script run_val.sh then 
```
bash run_val.sh
```
To run with the preprocessing edit the correct paths to your dataset and prediction output folder in the bash script run_val_dolp.sh then 
```
bash run_val_dolp.sh
```

## Results
Once you have run all the evaluations successfully you can edit paths in the bash script get_results.sh. (To replicate the results of the report you can download the outputs file above)
```
bash get_results.sh
```

## Acknowledgements
Huge thanks to [SAM6D](https://github.com/JiehongLin/SAM-6D/tree/main) and the [bpc_baseline](https://github.com/CIRP-Lab/bpc_baseline/tree/blog)


