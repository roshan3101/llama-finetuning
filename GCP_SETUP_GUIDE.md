# GCP Setup Guide for Training 8B Model

Complete guide for setting up and training the 8B model on Google Cloud Platform (first time).

## Prerequisites

- Google Cloud account (sign up at https://cloud.google.com/)
- Credit card (for billing, but you can use free credits)
- Basic familiarity with command line

## Step 1: Create GCP Project

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create a new project**:
   - Click "Select a project" → "New Project"
   - Name: `llama-finetuning` (or your choice)
   - Click "Create"
3. **Enable billing**:
   - Go to "Billing" in the menu
   - Link a billing account (you'll get $300 free credits for new accounts)

## Step 2: Enable Required APIs

1. **Go to "APIs & Services" → "Library"**
2. **Enable these APIs**:
   - Compute Engine API
   - Cloud Storage API (optional, for storing models)

## Step 3: Create VM Instance

### Option A: Using Console (Easier for First Time)

1. **Go to "Compute Engine" → "VM instances"**
2. **Click "Create Instance"**

**Configuration:**
- **Name**: `llama-training-vm`
- **Region**: `us-central1` or `us-east1` (cheaper)
- **Zone**: Any (e.g., `us-central1-a`)

**Machine Configuration:**
- **Machine family**: `N1` or `N2`
- **Machine type**: 
  - **Minimum**: `n1-standard-4` with GPU
  - **Recommended**: `n1-standard-8` with GPU
  - **Best**: `n1-highmem-8` with GPU (more RAM)

**GPU Configuration:**
- **GPU type**: `NVIDIA T4` (16GB VRAM) - **REQUIRED for 8B model**
- **Number of GPUs**: `1` (or `2` if you want faster training)
- **Note**: T4 is cheaper, A100 is faster but more expensive

**Boot Disk:**
- **OS**: `Ubuntu 22.04 LTS`
- **Disk size**: `100 GB` (minimum, 200GB recommended)
- **Disk type**: `Standard persistent disk` (or `SSD persistent disk` for faster I/O)

**Firewall:**
- ✅ Allow HTTP traffic
- ✅ Allow HTTPS traffic

3. **Click "Create"**

**Cost Estimate:**
- T4 GPU: ~$0.35/hour
- n1-standard-8: ~$0.38/hour
- **Total: ~$0.73/hour** (~$17.50/day)
- **With free credits, you can train for ~17 days!**

### Option B: Using gcloud CLI (Advanced)

```bash
gcloud compute instances create llama-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True
```

## Step 4: Connect to VM

### Option A: Using Browser SSH (Easiest)

1. **In GCP Console**, go to "Compute Engine" → "VM instances"
2. **Click "SSH"** next to your VM instance
3. Browser window opens with terminal

### Option B: Using gcloud CLI

```bash
gcloud compute ssh llama-training-vm --zone=us-central1-a
```

## Step 5: Set Up VM Environment

Once connected via SSH, run these commands:

### 5.1 Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 5.2 Install Python and Dependencies

```bash
# Install Python 3.10+
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Verify CUDA/GPU
nvidia-smi
# Should show your GPU (T4 or A100)
```

### 5.3 Install CUDA Toolkit (if needed)

```bash
# Check if CUDA is installed
nvcc --version

# If not installed, install CUDA 11.8 (compatible with PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 5.4 Clone Your Repository

```bash
# Option 1: If your code is on GitHub
git clone https://github.com/your-username/Career_guidance.git
cd Career_guidance

# Option 2: Upload files manually (see Step 6)
```

### 5.5 Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv
```

### 5.6 Set Up Hugging Face Token

```bash
# Create .env file
nano .env
# Add: HF_TOKEN=your_token_here
# Save: Ctrl+X, Y, Enter

# Or set environment variable
export HF_TOKEN="your_token_here"
```

## Step 6: Upload Your Code

### Option A: Using GitHub (Recommended)

If your code is on GitHub:
```bash
git clone https://github.com/your-username/Career_guidance.git
cd Career_guidance
```

### Option B: Using gcloud CLI (from your local machine)

```bash
# From your local machine
gcloud compute scp --recurse ./Career_guidance llama-training-vm:~/ --zone=us-central1-a
```

### Option C: Using Cloud Storage

1. **Upload to Cloud Storage** (from local machine):
```bash
gsutil -m cp -r ./Career_guidance gs://your-bucket-name/
```

2. **Download on VM**:
```bash
gsutil -m cp -r gs://your-bucket-name/Career_guidance ./
```

## Step 7: Prepare Data (if not already done)

```bash
cd Career_guidance
source venv/bin/activate

# Run data preparation
python scripts/prepare_data.py
```

**Note**: This will download datasets. May take 30-60 minutes.

## Step 8: Configure for 8B Model

The configuration is already set up! Just use `--model_size 8B`:

```bash
python training/train.py --model_size 8B --output_dir ./outputs
```

## Step 9: Start Training

### Basic Training

```bash
# Activate virtual environment
source venv/bin/activate

# Start training
python training/train.py --model_size 8B --output_dir ./outputs
```

### Training in Background (Recommended)

Since training takes hours/days, run it in background:

```bash
# Using nohup
nohup python training/train.py --model_size 8B --output_dir ./outputs > training.log 2>&1 &

# Or using screen (better)
screen -S training
python training/train.py --model_size 8B --output_dir ./outputs
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Monitor Training

```bash
# View logs
tail -f training.log

# Or if using screen
screen -r training

# Check GPU usage
watch -n 1 nvidia-smi
```

## Step 10: Save Your Work

### Download Results

From your **local machine**:

```bash
# Download outputs
gcloud compute scp --recurse llama-training-vm:~/Career_guidance/outputs ./outputs --zone=us-central1-a

# Download logs
gcloud compute scp --recurse llama-training-vm:~/Career_guidance/logs ./logs --zone=us-central1-a
```

### Or Upload to Cloud Storage

On VM:
```bash
# Install gsutil if needed
pip install gsutil

# Upload outputs
gsutil -m cp -r ./outputs gs://your-bucket-name/outputs/
```

## Step 11: Stop/Delete VM (Save Money!)

**IMPORTANT**: VMs cost money even when stopped (disk storage). Delete when done!

### Stop VM (keeps disk, costs less)

```bash
# From GCP Console: Compute Engine → VM instances → Stop
# Or from CLI:
gcloud compute instances stop llama-training-vm --zone=us-central1-a
```

### Delete VM (saves all money, but deletes everything)

```bash
# From GCP Console: Compute Engine → VM instances → Delete
# Or from CLI:
gcloud compute instances delete llama-training-vm --zone=us-central1-a
```

**Before deleting, make sure you've downloaded:**
- ✅ Model checkpoints (`./outputs/`)
- ✅ Training logs (`./logs/`)
- ✅ Any other important files

## Cost Optimization Tips

1. **Use Preemptible VMs** (60-80% cheaper, but can be terminated):
   - Add `--preemptible` flag when creating VM
   - Save checkpoints frequently!

2. **Use Spot VMs** (even cheaper):
   - Similar to preemptible, but different pricing

3. **Stop VM when not training**:
   - Stop VM when taking breaks
   - Only pay for disk storage (~$0.04/GB/month)

4. **Use appropriate instance size**:
   - Don't over-provision (8B model works on n1-standard-8)
   - Monitor GPU utilization

5. **Set up billing alerts**:
   - Go to "Billing" → "Budgets & alerts"
   - Set budget limit (e.g., $50)
   - Get notified before spending too much

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU
nvidia-smi

# If not working, install drivers
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Out of Memory

- Reduce batch size in `config/training_config.py`
- Increase gradient accumulation steps
- Use gradient checkpointing (already enabled)

### Connection Lost

- Use `screen` or `tmux` for persistent sessions
- Training will continue even if you disconnect

### Slow Training

- Check GPU utilization: `nvidia-smi`
- Should be >90% for good performance
- If low, check data loading (might be I/O bottleneck)

## Quick Start Script

Save this as `setup_gcp.sh` and run on VM:

```bash
#!/bin/bash
set -e

echo "Setting up GCP VM for 8B model training..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python
sudo apt-get install -y python3.10 python3.10-venv python3-pip git screen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv

# Verify GPU
echo "Checking GPU..."
nvidia-smi

echo "Setup complete! Next steps:"
echo "1. Set HF_TOKEN: export HF_TOKEN='your_token'"
echo "2. Prepare data: python scripts/prepare_data.py"
echo "3. Start training: python training/train.py --model_size 8B --output_dir ./outputs"
```

## Estimated Costs

**For 8B Model Training:**

- **VM**: n1-standard-8 with T4 GPU
- **Hourly cost**: ~$0.73/hour
- **Training time**: ~12-24 hours (3 epochs, 105K examples)
- **Total cost**: ~$9-18 (well within free $300 credit!)

**With free credits, you can train multiple times!**

## Next Steps After Training

1. **Download model**: Use `gcloud compute scp` or Cloud Storage
2. **Evaluate**: Run evaluation scripts
3. **Merge LoRA**: Merge adapters with base model
4. **Export**: Push to Hugging Face Hub
5. **Delete VM**: Save money!

## Support

- **GCP Documentation**: https://cloud.google.com/compute/docs
- **GPU Pricing**: https://cloud.google.com/compute/gpus-pricing
- **Free Tier**: https://cloud.google.com/free

---

**Good luck with your training!** 🚀
