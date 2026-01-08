# Google Colab Pro - 8B Model Setup

Quick setup guide specifically for training the 8B model on Google Colab Pro.

## 🚀 Quick Start for 8B Model

### Step 1: Open Colab and Setup

```python
# Cell 1: Mount Drive and Setup
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
from pathlib import Path

project_root = "/content/drive/MyDrive/Career_guidance"
Path(project_root).mkdir(parents=True, exist_ok=True)
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"✓ Working in: {os.getcwd()}")
```

### Step 2: Install Dependencies

```python
# Cell 2: Install packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard
```

### Step 3: Set Hugging Face Token

```python
# Cell 3: Configure HF
import os
os.environ['HF_TOKEN'] = 'your_token_here'  # ⚠️ Replace

from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
```

### Step 4: Verify GPU (Important for 8B!)

```python
# Cell 4: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# For 8B model, you need at least 16GB VRAM
# T4 (16GB) - will work but tight
# V100/A100 (32GB+) - recommended
```

### Step 5: Pre-Flight Check

```python
# Cell 5: Verify setup
!python scripts/preflight_check.py --model_size 8B
```

### Step 6: Prepare Data

```python
# Cell 6: Prepare training data
!python scripts/prepare_data.py
```

### Step 7: Train 8B Model

```python
# Cell 7: Start training
!python training/train.py --model_size 8B --output_dir ./outputs
```

**Training time:** 4-10 hours depending on GPU

## ⚙️ 8B Model Configuration

### Automatic Configuration

The pipeline automatically configures for 8B:
- Batch size: 2 per device
- Gradient accumulation: 8 steps
- Effective batch size: 16
- Sequence length: 1536 tokens
- LoRA rank: 32, alpha: 64

### If You Get Out of Memory (OOM)

**For T4 (16GB) - Reduce further:**

Edit `config/training_config.py` or create a custom config:

```python
# In Colab notebook, before training:
from config.training_config import TrainingConfig

# Create custom config for tight memory
custom_config = TrainingConfig()
custom_config.per_device_train_batch_size = 1
custom_config.gradient_accumulation_steps = 16
custom_config.max_seq_length = 1024
custom_config.output_dir = "./outputs"

# Then modify train.py to use this config
```

Or use environment variable override (if implemented).

## 📊 Monitoring 8B Training

### Check GPU Memory Usage

```python
# Monitor GPU during training
!nvidia-smi -l 1
```

**Expected usage for 8B:**
- T4 (16GB): ~14-15GB used (tight)
- V100 (32GB): ~20-25GB used (comfortable)
- A100 (40GB+): ~25-30GB used (plenty of headroom)

### View Training Progress

```python
# Watch logs
!tail -f logs/training_*.log
```

### TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir ./outputs
```

## 💾 Saving Checkpoints (8B Model)

**8B model checkpoints are larger:**
- LoRA adapters: ~32MB per checkpoint
- Full model (if merged): ~16GB

**Storage on Drive:**
- Each checkpoint: ~32MB
- Last 3 checkpoints kept: ~96MB total
- Final model: ~32MB (LoRA) or ~16GB (merged)

**Make sure you have Drive space!**

## 🔄 Resume Training

If Colab disconnects:

```python
# Resume from checkpoint
!python training/train.py \
    --model_size 8B \
    --output_dir ./outputs \
    --resume_from_checkpoint ./outputs/checkpoint-1000
```

## ✅ Complete Colab Notebook for 8B

```python
# ============================================
# Complete 8B Training Workflow
# ============================================

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')
import os, sys
from pathlib import Path
os.chdir("/content/drive/MyDrive/Career_guidance")
sys.path.insert(0, os.getcwd())

# 2. Install
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard

# 3. HF Token
import os
os.environ['HF_TOKEN'] = 'your_token'
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])

# 4. Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 5. Pre-flight
!python scripts/preflight_check.py --model_size 8B

# 6. Prepare data
!python scripts/prepare_data.py

# 7. Train
!python training/train.py --model_size 8B --output_dir ./outputs

# 8. Evaluate (after training)
!python scripts/evaluate_model.py \
    --model_path ./outputs \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --model_size 8B

# 9. Merge (optional)
!python export/merge_lora.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_adapter_path ./outputs \
    --output_path ./outputs/merged_model \
    --model_size 8B
```

## ⚠️ Important Notes for 8B

1. **GPU Requirements:**
   - Minimum: 16GB VRAM (T4 - tight)
   - Recommended: 32GB+ VRAM (V100/A100)

2. **Training Time:**
   - T4: 6-10 hours
   - V100: 4-6 hours
   - A100: 3-5 hours

3. **Memory Management:**
   - Default config is conservative
   - Reduce batch size if OOM
   - Increase gradient accumulation to compensate

4. **Checkpoints:**
   - Saved every 500 steps
   - Each checkpoint: ~32MB
   - Keep last 3 automatically

5. **Drive Storage:**
   - Ensure you have enough Drive space
   - 8B model files are larger than 1B

## 🎯 Quick Commands for 8B

```python
# Check setup
!python scripts/preflight_check.py --model_size 8B

# Prepare data
!python scripts/prepare_data.py

# Train
!python training/train.py --model_size 8B --output_dir ./outputs

# Resume
!python training/train.py --model_size 8B --output_dir ./outputs --resume_from_checkpoint ./outputs/checkpoint-1000

# Evaluate
!python scripts/evaluate_model.py --model_path ./outputs --base_model_path meta-llama/Llama-3.1-8B-Instruct --model_size 8B
```

---

**Ready to train 8B model on Colab Pro!** 🚀
