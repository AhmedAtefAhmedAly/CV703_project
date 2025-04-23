#!/bin/bash

# Define the URLs for the datasets
DATASET_URLS=(
    "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_models.zip"
    "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_val.zip"
    "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_train_pbr.zip"
)
# docker run --network=host -e BOP_PATH=/datasets/ -e DATASET_NAME=ipd -v /home/akasha_uswest2/bin_picking_challenge/ipd_codebase/datasets/:/datasets/ -it ibpc:tester
# Define the target directory
TARGET_DIR="./datasets"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Function to download and extract a dataset
download_and_extract() {
    local url="$1"
    local filename="$2"
    
    echo "Downloading $filename..."
    wget -O "$TARGET_DIR/$filename" "$url"
    
    # Check if the download was successful
    if [ $? -ne 0 ]; then
        echo "Download failed for $filename. Please check your internet connection or the URL."
        exit 1
    fi
    
    echo "Extracting $filename to $TARGET_DIR..."
    unzip -q "$TARGET_DIR/$filename" -d "$TARGET_DIR"
    
    # Remove the zip file after extraction
    rm "$TARGET_DIR/$filename"
    echo "Extraction complete for $filename."
}

# Loop through the dataset URLs and download/extract each one
for url in "${DATASET_URLS[@]}"; do
    filename=$(basename "$url")  # Extract the filename from the URL
    download_and_extract "$url" "$filename"
done

echo "All downloads and extractions complete. Datasets are ready in $TARGET_DIR."
