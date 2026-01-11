# Guide: Pushing Model to Hugging Face Hub

This guide explains everything you need to push your fine-tuned model to Hugging Face Hub.

## 📋 Requirements Checklist

### 1. **Hugging Face Account & Token** ✅
   - **What**: Authentication token to access Hugging Face Hub
   - **Where to get it**:
     1. Go to https://huggingface.co/settings/tokens
     2. Click "New token"
     3. Choose "Write" permissions (required for uploading)
     4. Copy the token (starts with `hf_`)
   - **How to use**: 
     - Set as environment variable: `HF_TOKEN=your_token_here`
     - Or pass directly: `--token your_token_here`
     - Or add to `.env` file: `HF_TOKEN=your_token_here`

### 2. **Trained Model Files** ✅
   - **What**: Your fine-tuned model (either LoRA adapters or merged model)
   - **Where to find**:
     - **LoRA adapters**: `./outputs/checkpoint-XXXX/` (from training)
     - **Merged model**: `./outputs/merged_model/` (after merging)
   - **Required files**:
     - `adapter_model.safetensors` or `adapter_model.bin` (LoRA weights)
     - `adapter_config.json` (LoRA configuration)
     - `tokenizer.json`, `tokenizer_config.json` (tokenizer files)
     - `README.md` (optional but recommended)
     - `training_args.bin` (optional, training configuration)

### 3. **Repository Name** ✅
   - **What**: Your Hugging Face repository ID
   - **Format**: `username/model-name` or `organization/model-name`
   - **Examples**:
     - `your-username/llama-8b-career-guidance`
     - `your-company/emotional-support-llama`
   - **Rules**:
     - Use lowercase letters, numbers, and hyphens
     - No spaces or special characters
     - Must be unique on Hugging Face

### 4. **Model Files (for merged model)** ✅
   - If pushing merged model, you need:
     - `model.safetensors` or `pytorch_model.bin` (model weights)
     - `config.json` (model configuration)
     - `tokenizer.json`, `tokenizer_config.json` (tokenizer)
     - `generation_config.json` (optional)

---

## 🚀 Step-by-Step Process

### Option 1: Push LoRA Adapters (Recommended - Smaller Size)

LoRA adapters are much smaller (~100MB-1GB) and can be loaded on top of the base model.

```bash
# 1. Make sure you have a checkpoint
ls ./outputs/checkpoint-7000/

# 2. Push LoRA adapters
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/llama-8b-career-guidance \
    --token hf_your_token_here
```

**What gets uploaded:**
- `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json` (LoRA config)
- `tokenizer.json`, `tokenizer_config.json`
- `README.md` (if present)

### Option 2: Push Merged Model (Full Model)

Merged model includes all weights and can be used standalone.

```bash
# 1. First, merge LoRA adapters with base model
python export/merge_lora.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_adapter_path ./outputs/checkpoint-7000 \
    --output_path ./outputs/merged_model \
    --model_size 8B

# 2. Push merged model
python export/push_to_huggingface.py \
    --model_path ./outputs/merged_model \
    --repo_id your-username/llama-8b-career-guidance-merged \
    --token hf_your_token_here
```

**What gets uploaded:**
- `model.safetensors` or `pytorch_model.bin` (full model weights)
- `config.json` (model configuration)
- `tokenizer.json`, `tokenizer_config.json`
- `generation_config.json` (if present)

---

## 📝 Detailed Instructions

### Step 1: Get Your Hugging Face Token

1. **Sign up/Login**: Go to https://huggingface.co/join
2. **Create Token**:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "llama-finetuning")
   - Select **"Write"** permissions (required for uploads)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again!)

3. **Save Token**:
   ```bash
   # Option A: Add to .env file
   echo "HF_TOKEN=hf_your_token_here" >> .env
   
   # Option B: Set environment variable
   export HF_TOKEN=hf_your_token_here
   
   # Option C: Pass directly in command (less secure)
   python export/push_to_huggingface.py --token hf_your_token_here ...
   ```

### Step 2: Choose Your Model Path

**For LoRA Adapters** (smaller, recommended):
```bash
MODEL_PATH="./outputs/checkpoint-7000"  # Latest checkpoint
```

**For Merged Model** (full model):
```bash
# First merge, then use merged path
MODEL_PATH="./outputs/merged_model"
```

### Step 3: Choose Repository Name

- **Format**: `username/model-name`
- **Examples**:
  - `john-doe/llama-8b-emotional-support`
  - `mycompany/career-guidance-llama`
- **Check availability**: The script will create it if it doesn't exist

### Step 4: Run Push Command

**Basic command:**
```bash
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/llama-8b-career-guidance \
    --token hf_your_token_here
```

**With environment variable (recommended):**
```bash
export HF_TOKEN=hf_your_token_here
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/llama-8b-career-guidance
```

**Make repository public:**
```bash
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/llama-8b-career-guidance \
    --public
```

---

## 📦 What Files Are Required?

### For LoRA Adapters (Minimum):
```
checkpoint-7000/
├── adapter_model.safetensors    # LoRA weights (REQUIRED)
├── adapter_config.json          # LoRA config (REQUIRED)
├── tokenizer.json               # Tokenizer (REQUIRED)
├── tokenizer_config.json        # Tokenizer config (REQUIRED)
└── README.md                    # Optional but recommended
```

### For Merged Model (Minimum):
```
merged_model/
├── model.safetensors            # Model weights (REQUIRED)
├── config.json                  # Model config (REQUIRED)
├── tokenizer.json               # Tokenizer (REQUIRED)
├── tokenizer_config.json        # Tokenizer config (REQUIRED)
└── generation_config.json       # Optional
```

---

## 🔍 Verification

After pushing, verify your model:

1. **Check on Hugging Face**:
   - Go to https://huggingface.co/your-username/your-model-name
   - You should see all uploaded files

2. **Test Loading**:
   ```python
   from peft import PeftModel, PeftConfig
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   # For LoRA adapters
   base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
   model = PeftModel.from_pretrained(base_model, "your-username/llama-8b-career-guidance")
   
   # For merged model
   model = AutoModelForCausalLM.from_pretrained("your-username/llama-8b-career-guidance-merged")
   ```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Token not found"
**Solution**: 
- Set `HF_TOKEN` environment variable
- Or pass `--token` argument
- Or add to `.env` file

### Issue 2: "Repository already exists"
**Solution**: 
- This is fine! The script will upload to existing repo
- Or use a different `--repo_id`

### Issue 3: "Permission denied" or "403 Forbidden - don't have the rights"
**Solution**: 
- **If using organization name** (e.g., `fixacity-roshan/model-name`):
  - Your token doesn't have permission to create repos in that organization
  - **Use your personal username instead**: `YOUR-USERNAME/model-name`
  - Or get added as a member of the organization
  - Or create the repo manually at https://huggingface.co/new
- **If using personal username**:
  - Make sure your token has **"Write"** permissions
  - Create a new token with write access at https://huggingface.co/settings/tokens
  - Check your username at https://huggingface.co/settings/profile

### Issue 4: "403 Forbidden - You don't have the rights to create a model under the namespace"
**Solution**: 
- **This means you're trying to create a repo under an organization you don't have access to**
- **Quick fix**: Use your personal username:
  ```bash
  # Instead of: fixacity-roshan/model-name
  # Use: YOUR-USERNAME/model-name
  python export/push_to_huggingface.py \
      --model_path ./outputs/merged_model \
      --repo_id YOUR-USERNAME/llama3.1-8b-emotional-career
  ```
- **To find your username**: Go to https://huggingface.co/settings/profile
- **Alternative**: Create the repo manually at https://huggingface.co/new, then run the push script

### Issue 5: "404 Not Found - Repository Not Found"
**Solution**: 
- Repository doesn't exist and couldn't be created (permission issue)
- Create it manually at https://huggingface.co/new
- Or use your personal username in the repo_id

### Issue 6: "Model path not found"
**Solution**: 
- Check the path exists: `ls ./outputs/checkpoint-7000`
- Use absolute path if needed
- Make sure checkpoint completed successfully

### Issue 7: "Upload failed - network error"
**Solution**: 
- Check internet connection
- Large models may take time - be patient
- Try again (uploads are resumable)

---

## 📊 File Size Estimates

| Model Size | LoRA Adapters | Merged Model |
|------------|---------------|--------------|
| 1B         | ~50-100 MB    | ~2-3 GB      |
| 8B         | ~200-500 MB   | ~15-20 GB    |
| 70B        | ~1-2 GB       | ~140-160 GB  |

**Note**: LoRA adapters are much smaller and recommended for most use cases.

---

## 🎯 Quick Reference

```bash
# 1. Set token (one-time setup)
export HF_TOKEN=hf_your_token_here

# 2. Push LoRA adapters (recommended)
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/your-model-name

# 3. Or merge first, then push
python export/merge_lora.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_adapter_path ./outputs/checkpoint-7000 \
    --output_path ./outputs/merged_model \
    --model_size 8B

python export/push_to_huggingface.py \
    --model_path ./outputs/merged_model \
    --repo_id your-username/your-model-name-merged
```

---

## 📚 Additional Resources

- **Hugging Face Hub Docs**: https://huggingface.co/docs/hub/index
- **Token Management**: https://huggingface.co/settings/tokens
- **Repository Creation**: https://huggingface.co/new
- **Model Cards Guide**: https://huggingface.co/docs/hub/model-cards

---

## ✅ Summary

**What you need:**
1. ✅ Hugging Face account + Write token
2. ✅ Trained model checkpoint (LoRA or merged)
3. ✅ Repository name (username/model-name)

**Where to get them:**
1. Token: https://huggingface.co/settings/tokens
2. Model: `./outputs/checkpoint-XXXX/` or `./outputs/merged_model/`
3. Repo name: Choose your own (must be unique)

**Command:**
```bash
python export/push_to_huggingface.py \
    --model_path ./outputs/checkpoint-7000 \
    --repo_id your-username/your-model-name
```

That's it! Your model will be available at:
`https://huggingface.co/your-username/your-model-name`
