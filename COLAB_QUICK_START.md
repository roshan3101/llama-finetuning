# Colab Quick Start - 8B Model

## 🚀 Fast Setup (Copy-Paste Ready)

### Complete Notebook Cells

**Cell 1: Setup**
```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys
from pathlib import Path
os.chdir("/content/drive/MyDrive/Career_guidance")
sys.path.insert(0, os.getcwd())
print(f"✓ {os.getcwd()}")
```

**Cell 2: Install**
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv tensorboard
```

**Cell 3: HF Token**
```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'  # ⚠️ REPLACE
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
```

**Cell 4: Check GPU**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Need 16GB+ for 8B model
```

**Cell 5: Pre-flight Check**
```python
!python scripts/preflight_check.py --model_size 8B
```

**Cell 6: Prepare Data**
```python
!python scripts/prepare_data.py
```

**Cell 7: Train 8B Model**
```python
!python training/train.py --model_size 8B --output_dir ./outputs
```

**Cell 8: Evaluate (After Training)**
```python
!python scripts/evaluate_model.py \
    --model_path ./outputs \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --model_size 8B
```

## 📋 What Gets Saved

All files automatically save to Google Drive:
- `outputs/checkpoint-XXX/` - Training checkpoints
- `outputs/adapter_model.safetensors` - Final LoRA adapters
- `outputs/merged_model/` - Merged model (after step 9)
- `logs/training_*.log` - Training logs

## ⚙️ 8B Model Settings

**Automatic configuration:**
- Batch size: 2 per device
- Gradient accumulation: 8 steps
- Effective batch: 16
- Sequence length: 1536 tokens
- LoRA: r=32, alpha=64

**If OOM error:**
- Reduce batch to 1: Edit `config/training_config.py`
- Or use 1B model instead

## ⏱️ Expected Times

- **Data prep**: 20-30 min (first time)
- **Training 8B**: 4-10 hours (depends on GPU)
- **Evaluation**: 5-10 minutes
- **Merge LoRA**: 10-20 minutes

## ✅ Success Checklist

- [ ] Drive mounted
- [ ] Dependencies installed
- [ ] HF token set
- [ ] Model access granted (8B)
- [ ] GPU has 16GB+ VRAM
- [ ] Data prepared
- [ ] Training started
- [ ] Checkpoints saving to Drive

---

**That's it!** Everything saves to Drive automatically. 🎉
