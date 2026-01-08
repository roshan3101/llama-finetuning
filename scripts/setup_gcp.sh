#!/bin/bash
# Setup script for GCP VM - Run this after connecting to your VM

set -e

echo "=========================================="
echo "Setting up GCP VM for 8B Model Training"
echo "=========================================="

# Update system
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
echo "Step 2: Installing Python and dependencies..."
sudo apt-get install -y python3.10 python3.10-venv python3-pip git screen htop

# Verify GPU
echo "Step 3: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ GPU detected!"
else
    echo "⚠ GPU drivers not installed. Installing..."
    sudo apt-get install -y nvidia-driver-535
    echo "⚠ Please reboot after installation: sudo reboot"
fi

# Create virtual environment
echo "Step 4: Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
echo "Step 5: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo "Step 6: Installing project dependencies..."
pip install transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv

# Verify installations
echo "Step 7: Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set Hugging Face token:"
echo "   export HF_TOKEN='your_token_here'"
echo "   # Or create .env file: echo 'HF_TOKEN=your_token' > .env"
echo ""
echo "2. Upload your code (if not already done):"
echo "   # Option 1: git clone https://github.com/your-username/Career_guidance.git"
echo "   # Option 2: Use gcloud compute scp from local machine"
echo ""
echo "3. Prepare data:"
echo "   cd Career_guidance"
echo "   source venv/bin/activate"
echo "   python scripts/prepare_data.py"
echo ""
echo "4. Start training (in screen session):"
echo "   screen -S training"
echo "   python training/train.py --model_size 8B --output_dir ./outputs"
echo "   # Detach: Ctrl+A, then D"
echo "   # Reattach: screen -r training"
echo ""
echo "5. Monitor training:"
echo "   tail -f training.log"
echo "   watch -n 1 nvidia-smi"
echo ""
