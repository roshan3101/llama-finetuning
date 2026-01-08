#!/bin/bash
# Download training results from GCP VM to local machine
# Run this from your LOCAL machine, not on the VM

set -e

VM_NAME=${1:-llama-training-vm}
ZONE=${2:-us-central1-a}
LOCAL_DIR=${3:-./gcp_outputs}

echo "=========================================="
echo "Downloading Results from GCP VM"
echo "=========================================="
echo "VM Name: $VM_NAME"
echo "Zone: $ZONE"
echo "Local Directory: $LOCAL_DIR"
echo ""

# Create local directory
mkdir -p "$LOCAL_DIR"

# Download outputs
echo "Downloading outputs..."
gcloud compute scp --recurse \
    $VM_NAME:~/Career_guidance/outputs \
    "$LOCAL_DIR/outputs" \
    --zone=$ZONE

# Download logs
echo "Downloading logs..."
gcloud compute scp --recurse \
    $VM_NAME:~/Career_guidance/logs \
    "$LOCAL_DIR/logs" \
    --zone=$ZONE

# Download data (optional)
read -p "Download processed data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading processed data..."
    gcloud compute scp --recurse \
        $VM_NAME:~/Career_guidance/data/processed \
        "$LOCAL_DIR/data/processed" \
        --zone=$ZONE
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Results saved to: $LOCAL_DIR"
echo ""
