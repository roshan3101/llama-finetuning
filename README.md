# LLaMA 3.1 Fine-Tuning Pipeline

Production-grade fine-tuning pipeline for LLaMA 3.1 models focused on emotional support and career guidance. Uses QLoRA (4-bit quantization + LoRA) for efficient fine-tuning.

## Features

- **Modular Architecture**: Clean separation of concerns (config, data, training, evaluation, export)
- **Scalable Design**: Easy to scale from 1B to 70B models
- **Safety First**: Built-in filtering for medical advice and unsafe content
- **Production Ready**: Comprehensive logging, error handling, and evaluation
- **QLoRA Efficient**: 4-bit quantization with LoRA adapters for memory-efficient training

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── model_config.py     # Model and quantization settings
│   ├── training_config.py  # Training hyperparameters
│   └── paths_config.py     # Dataset and output paths
│
├── data/                   # Data processing
│   ├── raw/                # Raw datasets (downloaded)
│   ├── processed/          # Processed datasets (JSONL)
│   └── scripts/
│       ├── load_datasets.py      # Load datasets from HuggingFace
│       ├── clean_filter.py       # Safety and quality filtering
│       ├── format_instructions.py # Convert to instruction format
│       └── train_val_split.py    # Train/validation split
│
├── training/               # Training pipeline
│   ├── load_model.py       # Model and tokenizer loading
│   ├── apply_lora.py       # LoRA adapter setup
│   ├── trainer.py          # Custom trainer and data collator
│   └── train.py            # Main training script
│
├── evaluation/             # Model evaluation
│   ├── sample_generation.py    # Generate test samples
│   └── qualitative_checks.py   # Safety and quality checks
│
├── utils/                  # Utilities
│   ├── logging.py          # Logging configuration
│   ├── safety_filters.py   # Safety filtering functions
│   └── prompt_templates.py # Prompt templates
│
└── export/                 # Model export
    ├── merge_lora.py       # Merge LoRA with base model
    └── push_to_huggingface.py # Upload to HuggingFace Hub
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 4GB+ VRAM for 1B, 16GB+ for 8B, 40GB+ for 70B)
- Hugging Face account with access to LLaMA 3.1 models

### Setup

1. **Clone the repository** (or navigate to project directory)

2. **Install dependencies**:

```bash
pip install torch transformers datasets peft bitsandbytes accelerate huggingface-hub
```

3. **Set up Hugging Face token**:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

4. **Request access to LLaMA models** on Hugging Face:
   - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (for 1B)
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (for 8B)
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct (for 70B)
   - Request access and accept the license for your chosen model

## Usage

### 1. Data Preparation

#### Load and Process Datasets

Create a script to run the data pipeline:

```python
# scripts/prepare_data.py
from data.scripts.load_datasets import (
    load_all_emotional_support_datasets,
    load_all_career_guidance_datasets,
)
from data.scripts.format_instructions import format_dataset
from data.scripts.clean_filter import filter_dataset
from data.scripts.train_val_split import split_dataset, save_split
from config.paths_config import get_paths_config
import json

paths_config = get_paths_config()

# Load datasets
emotional_datasets = load_all_emotional_support_datasets()
career_datasets = load_all_career_guidance_datasets()

# Format and combine
all_examples = []

for name, dataset in emotional_datasets.items():
    examples = [ex for ex in dataset]
    formatted = format_dataset(examples, name)
    all_examples.extend(formatted)

for name, dataset in career_datasets.items():
    examples = [ex for ex in dataset]
    formatted = format_dataset(examples, name)
    all_examples.extend(formatted)

# Filter for safety
filtered = filter_dataset(all_examples, strict_filtering=True)

# Split train/val
train_examples, val_examples = split_dataset(
    filtered,
    train_ratio=0.9,
    seed=42
)

# Save
save_split(
    train_examples,
    val_examples,
    paths_config.train_dataset_path,
    paths_config.val_dataset_path
)

print(f"Prepared {len(train_examples)} train and {len(val_examples)} val examples")
```

Run:
```bash
python scripts/prepare_data.py
```

### 2. Training

#### Basic Training

```bash
python training/train.py \
    --model_size 13B \
    --output_dir ./outputs/llama3.1-13b-finetuned \
    --hf_token $HF_TOKEN
```

#### Resume from Checkpoint

```bash
python training/train.py \
    --model_size 1B \
    --output_dir ./outputs/llama3.2-1b-finetuned \
    --resume_from_checkpoint ./outputs/llama3.2-1b-finetuned/checkpoint-1000
```

#### Training on Google Colab

1. Upload the project to Google Drive or clone from GitHub
2. Install dependencies in Colab:
```python
!pip install torch transformers datasets peft bitsandbytes accelerate huggingface-hub
```
3. Set HF token:
```python
import os
os.environ["HF_TOKEN"] = "your_token"
```
4. Run training script (adjust paths as needed)

### 3. Evaluation

#### Generate Test Samples

```python
# scripts/evaluate_model.py
from evaluation.sample_generation import (
    load_model_for_inference,
    generate_samples,
    load_test_cases,
)
from config.model_config import get_model_config
from config.paths_config import get_paths_config

model_config = get_model_config("1B")
paths_config = get_paths_config()

# Load model (adjust paths)
model, tokenizer = load_model_for_inference(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    lora_adapter_path="./outputs/llama3.2-1b-finetuned",
    model_config=model_config,
)

# Load test cases
test_cases = load_test_cases("data/test_cases.jsonl")

# Generate samples
generate_samples(
    model=model,
    tokenizer=tokenizer,
    test_cases=test_cases,
    output_path=paths_config.eval_samples_path,
)
```

#### Run Qualitative Evaluation

```python
from evaluation.qualitative_checks import evaluate_samples
from config.paths_config import get_paths_config

paths_config = get_paths_config()

evaluate_samples(
    samples_path=paths_config.eval_samples_path,
    output_path=paths_config.qualitative_results_path,
)
```

### 4. Export

#### Merge LoRA Adapters

```bash
python export/merge_lora.py \
    --base_model_path meta-llama/Llama-3.2-1B-Instruct \
    --lora_adapter_path ./outputs/llama3.2-1b-finetuned \
    --output_path ./outputs/merged_model \
    --model_size 1B
```

#### Push to Hugging Face Hub

```bash
python export/push_to_huggingface.py \
    --model_path ./outputs/merged_model \
    --repo_id your-username/llama3.2-1b-emotional-career \
    --token $HF_TOKEN
```

## Configuration

### Model Configuration

Edit `config/model_config.py` to adjust:
- Model size (1B, 8B, 70B)
- LoRA rank and alpha
- Quantization settings
- Target modules

### Training Configuration

Edit `config/training_config.py` to adjust:
- Batch size and gradient accumulation
- Learning rate and scheduler
- Number of epochs (default: 2 epochs)
- Logging and checkpointing frequency (default: checkpoint every 100 steps for Colab stability)

### Paths Configuration

Edit `config/paths_config.py` to adjust:
- Dataset paths
- Output directories
- Hugging Face repository ID

## Scaling to Larger Models

### 1B → 8B → 70B

1. **Update model config**:
```python
model_config = get_model_config("70B")  # or "1B", "8B"
```

2. **Adjust training config**:
```python
training_config = get_training_config_for_model_size("70B", gpu_memory_gb=80)
```

3. **Use multi-GPU** (if available):
```bash
torchrun --nproc_per_node=4 training/train.py --model_size 70B
```

## Dataset Information

### Emotional Support Datasets

- **Empathetic Dialogues**: Conversational emotional support
- **Go Emotions**: Emotion classification (used for prompting)
- **Mental Health Counseling**: Requires strict filtering (no medical advice)

### Career Guidance Datasets

- **Career Guidance QA**: Question-answer pairs
- **Karrierewege**: Career trajectory data (synthetic generation)
- **ESCO Taxonomy**: Skills and occupations (for RAG, not fine-tuning)

## Safety and Filtering

The pipeline includes comprehensive safety filtering:

- **Medical Advice Detection**: Filters out clinical/medical content
- **Unsafe Content Detection**: Filters harmful or inappropriate content
- **Quality Checks**: Removes low-quality, repetitive, or too-short examples

All filtering is configurable in `data/scripts/clean_filter.py`.

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` in training config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_seq_length`
- Use gradient checkpointing (enabled by default)

### Slow Training

- **A100 GPU**: The pipeline automatically detects A100 (40GB+) and applies high-performance optimizations:
  - Large batch size (8 for 8B model vs 2 for T4)
  - bfloat16 precision (faster than fp16)
  - Optimized data loading (4 workers)
  - Full sequence length (2048 tokens)
  - Faster optimizer (adamw_torch)
  - **Result: 2-4x faster training!**
- For other GPUs: Increase `gradient_accumulation_steps` to use larger effective batch size
- Enable `bf16` if your GPU supports it (set `fp16=False`, `bf16=True`)
- Use `packing=True` for more efficient data loading (requires custom collator)

### Model Not Learning

- Check learning rate (try 1e-4 to 5e-4 for LoRA)
- Verify data format matches instruction template
- Check that LoRA adapters are being trained (see trainable parameters log)

## TODO / Manual Steps

1. **Dataset Access**: Some datasets may require manual download or access requests
2. **Karrierewege Formatting**: Implement based on actual dataset structure
3. **ESCO Integration**: Set up for RAG layer (not part of fine-tuning)
4. **Multi-GPU Training**: Configure DDP for multi-GPU setups
5. **Custom Metrics**: Add domain-specific evaluation metrics

## License

This pipeline is for internal company use. Ensure compliance with:
- LLaMA 3.1 model license (Meta)
- Dataset licenses (check each dataset's terms)
- Company policies on AI model deployment

## Support

For issues or questions:
1. Check logs in `./logs/`
2. Review configuration files
3. Verify dataset formats match expected structure
4. Check Hugging Face model access

---

**Note**: This pipeline focuses on training and evaluation only. Inference hosting and API deployment are separate concerns and not included in this codebase.

