# Post-Training Guide

## What Happens After Training Completes

When training finishes, here's what you'll have and what to do next:

## 📁 Files Created During Training

### 1. **Checkpoints** (Saved During Training)
- Location: `./outputs/checkpoint-XXX/` (where XXX is step number)
- Contains:
  - LoRA adapter weights (`adapter_model.safetensors`)
  - Training state (`training_state.json`)
  - Optimizer state
- **Keep the last 3 checkpoints** (configured in `training_config.py`)

### 2. **Final Model** (Saved After Training)
- Location: `./outputs/` (or your specified output_dir)
- Contains:
  - LoRA adapter weights (final trained adapters)
  - Tokenizer files
  - Configuration files
- **This is your fine-tuned model!**

### 3. **Training Logs**
- Location: `./logs/training_YYYYMMDD_HHMMSS.log`
- Contains: All training progress, losses, metrics

## 🔍 Step 1: Evaluate Your Model

### Quick Test

Test the model with sample questions:

```powershell
python scripts/evaluate_model.py --model_path ./outputs --base_model_path meta-llama/Llama-3.2-1B-Instruct --model_size 1B
```

**What this does:**
- Loads your fine-tuned model
- Generates responses to test questions
- Saves samples to `./outputs/eval_samples.jsonl`
- Runs qualitative checks (safety, empathy, professionalism)
- Saves evaluation results to `./outputs/qualitative_results.json`

**Expected output:**
- Sample responses from your model
- Safety scores
- Quality metrics
- Overall evaluation summary

### Manual Testing

You can also test interactively:

```python
from evaluation.sample_generation import load_model_for_inference, generate_response
from config.model_config import get_model_config

model_config = get_model_config("1B")
model, tokenizer = load_model_for_inference(
    base_model_path="meta-llama/Llama-3.2-1B-Instruct",
    lora_adapter_path="./outputs",
    model_config=model_config,
)

# Test a question
response = generate_response(
    model=model,
    tokenizer=tokenizer,
    instruction="I'm feeling stressed about my job interview tomorrow.",
    max_new_tokens=200,
    temperature=0.7
)
print(response)
```

## 🔄 Step 2: Merge LoRA Adapters (Optional but Recommended)

**Why merge?**
- LoRA adapters are small files that modify the base model
- Merged model is easier to use (no need to load base + adapters separately)
- Better for deployment and sharing

**How to merge:**

```powershell
python export/merge_lora.py `
    --base_model_path meta-llama/Llama-3.2-1B-Instruct `
    --lora_adapter_path ./outputs `
    --output_path ./outputs/merged_model `
    --model_size 1B
```

**What this does:**
- Loads base model
- Loads your LoRA adapters
- Merges them into a single model
- Saves to `./outputs/merged_model/`

**Result:**
- Full model file (larger, but self-contained)
- Can be used without PEFT library
- Easier to deploy

## 📤 Step 3: Export to Hugging Face Hub (Optional)

If you want to share or backup your model:

```powershell
python export/push_to_huggingface.py `
    --model_path ./outputs/merged_model `
    --repo_id your-username/llama3.2-1b-emotional-career `
    --token $env:HF_TOKEN
```

**What this does:**
- Creates a repository on Hugging Face (if it doesn't exist)
- Uploads your model files
- Makes it available for download/sharing

**Note:** 
- Repository will be **private** by default
- Add `--public` flag if you want it public
- You need write access to the repo (or create a new one)

## 📊 Step 4: Analyze Training Results

### Check Training Metrics

Look at your training log:
```powershell
# View latest log
Get-Content ./logs/training_*.log -Tail 50
```

**What to look for:**
- Training loss (should decrease over time)
- Evaluation loss (should decrease)
- Learning rate schedule
- Any errors or warnings

### Check Model Performance

Review evaluation results:
```powershell
# View evaluation results
Get-Content ./outputs/qualitative_results.json
```

**Metrics to check:**
- Safety rate (should be high, >95%)
- Empathy rate (should improve with training)
- Professionalism rate
- Overall quality score

## 🎯 Step 5: Iterate and Improve

### If Results Are Good ✅
- You're done! Use the model for inference
- Consider merging and exporting
- Document your model's capabilities

### If Results Need Improvement 🔄

**Options:**
1. **Train longer**: Increase epochs in `config/training_config.py`
2. **Adjust learning rate**: Try different values (1e-4 to 5e-4)
3. **More data**: Add more training examples
4. **Different LoRA settings**: Adjust `lora_r` and `lora_alpha` in `config/model_config.py`
5. **Resume training**: Continue from a checkpoint
   ```powershell
   python training/train.py --model_size 1B --output_dir ./outputs --resume_from_checkpoint ./outputs/checkpoint-1000
   ```

## 📝 Step 6: Document Your Model

Create a model card documenting:
- What the model does
- Training data used
- Performance metrics
- Limitations
- Usage instructions

## 🚀 Step 7: Use Your Model

### For Inference

Once you have your model (merged or LoRA), you can use it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained("./outputs/merged_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/merged_model")

# Or load LoRA model
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "./outputs")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Generate response
prompt = "### Instruction:\nI'm feeling anxious about my career change.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📂 File Structure After Training

```
outputs/
├── checkpoint-500/          # Intermediate checkpoint
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── training_state.json
├── checkpoint-1000/          # Another checkpoint
├── checkpoint-1500/          # Final checkpoint
├── adapter_model.safetensors  # Final LoRA adapters
├── adapter_config.json
├── tokenizer_config.json
├── tokenizer.json
└── training_state.json

merged_model/                 # After merging (optional)
├── model.safetensors        # Full merged model
├── config.json
└── tokenizer files

eval_samples.jsonl           # Evaluation samples
qualitative_results.json     # Evaluation metrics
```

## ⚠️ Important Notes

1. **LoRA vs Merged Model:**
   - LoRA adapters are small (~11MB for 1B model)
   - Merged model is full size (~2.5GB for 1B model)
   - Use LoRA for development, merged for deployment

2. **Checkpoints:**
   - Keep best checkpoint (lowest eval loss)
   - Can resume from any checkpoint
   - Old checkpoints are auto-deleted (keeps last 3)

3. **Model Size:**
   - 1B model: ~2.5GB merged, ~11MB LoRA
   - 8B model: ~16GB merged, ~32MB LoRA
   - 70B model: ~140GB merged, ~128MB LoRA

4. **Next Steps:**
   - Test thoroughly before deployment
   - Monitor for safety issues
   - Consider adding RAG layer for domain knowledge
   - Set up inference API (separate from this pipeline)

## 🎉 Success Checklist

After training, you should have:
- [ ] Trained LoRA adapters in `./outputs/`
- [ ] Evaluation results showing good performance
- [ ] (Optional) Merged model for easier deployment
- [ ] (Optional) Model uploaded to Hugging Face
- [ ] Documentation of model capabilities

## Quick Reference Commands

```powershell
# Evaluate model
python scripts/evaluate_model.py --model_path ./outputs --base_model_path meta-llama/Llama-3.2-1B-Instruct --model_size 1B

# Merge LoRA adapters
python export/merge_lora.py --base_model_path meta-llama/Llama-3.2-1B-Instruct --lora_adapter_path ./outputs --output_path ./outputs/merged_model --model_size 1B

# Push to Hugging Face
python export/push_to_huggingface.py --model_path ./outputs/merged_model --repo_id your-username/your-model-name --token $env:HF_TOKEN

# Resume training from checkpoint
python training/train.py --model_size 1B --output_dir ./outputs --resume_from_checkpoint ./outputs/checkpoint-1000
```

---

**Congratulations!** You've successfully fine-tuned a model for emotional support and career guidance! 🎊
