# Google Colab Pro Setup Guide

Complete guide for running the fine-tuning pipeline on Google Colab Pro.

## 🚀 Quick Start

### Step 1: Open Colab and Clone Repository

1. **Open Google Colab**: https://colab.research.google.com/
2. **Create a new notebook**
3. **Mount Google Drive** (for persistent storage):

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. **Clone or upload the repository**:

**Option A: Clone from GitHub** (if you've pushed to GitHub):
```python
!cd /content/drive/MyDrive && git clone https://github.com/your-username/llama-finetuning.git
```

**Option B: Upload files directly**:
- Upload the entire project folder to `/content/drive/MyDrive/llama-finetuning/`
- Or use Colab's file upload feature

5. **Navigate to project**:
```python
import os
os.chdir('/content/drive/MyDrive/llama-finetuning')
```

### Step 2: Run Setup Script

```python
# Run the Colab setup
exec(open('colab_setup.py').read())
```

Or run setup manually:

```python
# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard

# Set up paths
import os
from pathlib import Path

# Mount Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Set project root
project_root = "/content/drive/MyDrive/llama-finetuning"
os.chdir(project_root)

# Create directories
for dir_name in ["data/raw", "data/processed", "outputs", "logs"]:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

print(f"✓ Setup complete! Working in: {os.getcwd()}")
```

### Step 3: Set Hugging Face Token

```python
import os

# Set your Hugging Face token
os.environ['HF_TOKEN'] = 'hf_your_token_here'

# Login
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])

print("✓ Logged in to Hugging Face")
```

### Step 4: Verify GPU

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Expected output:**
- CUDA available: True
- GPU: T4, V100, or A100 (depending on Colab Pro tier)
- Memory: 15-40 GB

### Step 5: Run Pre-Flight Check

```python
!python scripts/preflight_check.py --model_size 1B
```

Should show all checks passing.

### Step 6: Prepare Data

```python
!python scripts/prepare_data.py
```

**Time:** 10-30 minutes (first time, downloads datasets)
**Output:** `data/processed/train.jsonl` and `val.jsonl`

### Step 7: Start Training

```python
!python training/train.py --model_size 1B --output_dir ./outputs
```

**Training time for 8B model:**
- Colab Pro T4 (16GB): 6-10 hours (may need batch size reduction)
- Colab Pro V100 (32GB): 4-6 hours
- Colab Pro+ A100 (40GB+): 3-5 hours

## 📁 Colab-Specific Configuration

### Paths Configuration

The code automatically detects Colab and adjusts paths. Files will be saved to:
- **With Drive mounted**: `/content/drive/MyDrive/llama-finetuning/outputs/`
- **Without Drive**: `/content/llama-finetuning/outputs/` (⚠️ lost when session ends)

### Persistent Storage

**Recommended:** Always mount Google Drive for:
- ✅ Model checkpoints persist
- ✅ Training logs saved
- ✅ Data cached
- ✅ Can resume training later

**Drive Mount:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Checkpoint Saving

Checkpoints are automatically saved to Drive (if mounted):
- Location: `/content/drive/MyDrive/llama-finetuning/outputs/checkpoint-XXX/`
- Saves every 500 steps (configurable)
- Keeps last 3 checkpoints

## 🔧 Colab-Specific Adjustments

### Memory Management for 8B Model

For 8B model on Colab, you may need to adjust:

```python
# In config/training_config.py, for 8B model:
# Colab Pro T4 (16GB) - may need:
per_device_train_batch_size = 1  # Reduce if OOM
gradient_accumulation_steps = 16  # Increase to maintain effective batch size
max_seq_length = 1024  # Reduce if needed

# Colab Pro V100/A100 (32GB+) - can use:
per_device_train_batch_size = 4  # Default for 8B
gradient_accumulation_steps = 4
max_seq_length = 2048
```

### GPU Selection

Colab Pro gives you better GPUs. Check what you got:

```python
!nvidia-smi
```

**GPU tiers for 8B model:**
- **Free**: T4 (16GB) - ⚠️ Tight, may need batch size reduction
- **Pro**: T4/V100 (16-32GB) - ✅ Good for 8B
- **Pro+**: A100 (40-80GB) - ✅ Excellent for 8B, can handle 70B

### Session Management

**Important:** Colab sessions can disconnect!

1. **Save checkpoints frequently** (already configured)
2. **Use Drive for storage** (persists across sessions)
3. **Resume from checkpoint** if session disconnects:

```python
!python training/train.py \
    --model_size 1B \
    --output_dir ./outputs \
    --resume_from_checkpoint ./outputs/checkpoint-1000
```

## 📊 Monitoring Training

### View Logs in Real-Time

```python
# Watch training log
!tail -f logs/training_*.log
```

### TensorBoard (Optional)

```python
# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./outputs
```

### Check GPU Usage

```python
# Monitor GPU
!nvidia-smi -l 1  # Updates every second
```

## 💾 Saving and Downloading

### Download Checkpoints

```python
from google.colab import files
import shutil

# Create zip of outputs
shutil.make_archive('model_checkpoint', 'zip', 'outputs')

# Download
files.download('model_checkpoint.zip')
```

### Save to Drive (Automatic)

If Drive is mounted, everything saves automatically to:
- `/content/drive/MyDrive/llama-finetuning/outputs/`

## 🔄 Complete Colab Workflow

### Full Notebook Cell Sequence

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Career_guidance')

# Cell 2: Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard

# Cell 3: Set token and login
import os
os.environ['HF_TOKEN'] = 'your_token_here'
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])

# Cell 4: Verify setup
!python scripts/preflight_check.py --model_size 1B

# Cell 5: Prepare data
!python scripts/prepare_data.py

# Cell 6: Start training
!python training/train.py --model_size 1B --output_dir ./outputs

# Cell 7: (After training) Evaluate
!python scripts/evaluate_model.py --model_path ./outputs --base_model_path meta-llama/Llama-3.2-1B-Instruct --model_size 1B

# Cell 8: (Optional) Merge LoRA
!python export/merge_lora.py --base_model_path meta-llama/Llama-3.2-1B-Instruct --lora_adapter_path ./outputs --output_path ./outputs/merged_model --model_size 1B

# Cell 9: (Optional) Push to Hugging Face
!python export/push_to_huggingface.py --model_path ./outputs/merged_model --repo_id your-username/your-model-name --token $env:HF_TOKEN
```

## ⚠️ Colab-Specific Issues

### Session Timeout

**Problem:** Colab disconnects after inactivity
**Solution:**
- Keep browser tab active
- Use `!python` commands (they keep session alive)
- Resume from checkpoint if disconnected

### Out of Memory (OOM)

**Problem:** GPU runs out of memory
**Solution:**
```python
# Reduce batch size in config/training_config.py
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
max_seq_length = 512  # Reduce sequence length
```

### Drive Quota

**Problem:** Google Drive storage full
**Solution:**
- Check Drive storage: https://drive.google.com/drive/quota
- Delete old checkpoints (keeps last 3 automatically)
- Use Colab's temporary storage for testing

### Slow Downloads

**Problem:** Dataset/model downloads are slow
**Solution:**
- First run downloads everything (cached after)
- Use Colab Pro for faster internet
- Pre-download to Drive if needed

## 📝 Colab Notebook Template

Create a new Colab notebook with these cells:

```python
# ============================================
# Cell 1: Setup and Mount Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
from pathlib import Path

# Set project root
project_root = "/content/drive/MyDrive/llama-finetuning"
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"✓ Working directory: {os.getcwd()}")

# ============================================
# Cell 2: Install Dependencies
# ============================================
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard

# ============================================
# Cell 3: Configure Hugging Face
# ============================================
import os
os.environ['HF_TOKEN'] = 'your_token_here'  # ⚠️ Replace with your token

from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])

# ============================================
# Cell 4: Verify Setup
# ============================================
!python scripts/preflight_check.py --model_size 1B

# ============================================
# Cell 5: Prepare Data
# ============================================
!python scripts/prepare_data.py

# ============================================
# Cell 6: Train Model
# ============================================
!python training/train.py --model_size 1B --output_dir ./outputs

# ============================================
# Cell 7: Evaluate Model
# ============================================
!python scripts/evaluate_model.py \
    --model_path ./outputs \
    --base_model_path meta-llama/Llama-3.2-1B-Instruct \
    --model_size 1B

# ============================================
# Cell 8: Merge LoRA (Optional)
# ============================================
!python export/merge_lora.py \
    --base_model_path meta-llama/Llama-3.2-1B-Instruct \
    --lora_adapter_path ./outputs \
    --output_path ./outputs/merged_model \
    --model_size 1B
```

## 🎯 Quick Reference

### Essential Commands

```python
# Check GPU
!nvidia-smi

# Check setup
!python scripts/preflight_check.py --model_size 8B

# Prepare data
!python scripts/prepare_data.py

# Train (8B model - requires Colab Pro with good GPU)
!python training/train.py --model_size 8B --output_dir ./outputs

# Resume training
!python training/train.py --model_size 8B --output_dir ./outputs --resume_from_checkpoint ./outputs/checkpoint-1000

# Evaluate
!python scripts/evaluate_model.py --model_path ./outputs --base_model_path meta-llama/Llama-3.1-8B-Instruct --model_size 8B
```

### File Locations

- **Project**: `/content/drive/MyDrive/llama-finetuning/`
- **Outputs**: `/content/drive/MyDrive/llama-finetuning/outputs/`
- **Logs**: `/content/drive/MyDrive/llama-finetuning/logs/`
- **Data**: `/content/drive/MyDrive/llama-finetuning/data/processed/`

## 💡 Tips for Colab

1. **Use Drive**: Always mount Drive for persistent storage
2. **Monitor GPU**: Keep `nvidia-smi` running to watch memory
3. **Save frequently**: Checkpoints save every 500 steps
4. **Resume capability**: Can resume from any checkpoint
5. **Session management**: Keep browser active to avoid disconnects
6. **Pro vs Free**: Pro gives better GPUs and longer sessions

## 🚨 Troubleshooting

### "ModuleNotFoundError"
- Run setup cell again
- Check you're in the right directory

### "Out of Memory"
- Reduce batch size
- Use smaller model (1B instead of 8B)
- Reduce max_seq_length

### "Session Disconnected"
- Resume from last checkpoint
- Files on Drive are safe

### "Drive Mount Failed"
- Check Drive permissions
- Try remounting
- Use temporary storage as fallback

---

**Ready to train on Colab!** Follow the steps above and your model will be saved to Google Drive automatically. 🚀
