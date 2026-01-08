"""
Google Colab Setup Script

Run this in a Colab notebook cell to set up the environment.
"""

import os
import sys
from pathlib import Path

def setup_colab():
    """Setup environment for Google Colab."""
    print("=" * 60)
    print("Setting up Fine-Tuning Pipeline for Google Colab")
    print("=" * 60)
    
    # Mount Google Drive
    print("\n1. Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted at /content/drive")
    except ImportError:
        print("⚠ Not running in Colab - skipping Drive mount")
        drive_mounted = False
    else:
        drive_mounted = True
    
    # Set project path
    if drive_mounted:
        # Option 1: Use Drive (persistent storage)
        project_root = "/content/drive/MyDrive/Career_guidance"
        print(f"✓ Using Drive path: {project_root}")
    else:
        # Option 2: Use Colab's /content (temporary)
        project_root = "/content/Career_guidance"
        print(f"⚠ Using temporary path: {project_root}")
    
    # Change to project directory
    os.chdir(project_root)
    print(f"✓ Changed to: {os.getcwd()}")
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install -q transformers datasets peft bitsandbytes accelerate huggingface-hub python-dotenv")
    os.system("pip install -q tensorboard")  # For training visualization
    print("✓ Dependencies installed")
    
    # Set up Hugging Face token
    print("\n3. Setting up Hugging Face token...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠ HF_TOKEN not set. Please set it:")
        print("   os.environ['HF_TOKEN'] = 'your_token_here'")
        print("   Or use huggingface-cli login")
    else:
        print("✓ Hugging Face token found")
    
    # Login to Hugging Face
    print("\n4. Logging in to Hugging Face...")
    try:
        from huggingface_hub import login
        if hf_token:
            login(token=hf_token)
            print("✓ Logged in to Hugging Face")
        else:
            print("⚠ Skipping login - set HF_TOKEN first")
    except Exception as e:
        print(f"⚠ Login failed: {e}")
    
    # Create necessary directories
    print("\n5. Creating directories...")
    directories = [
        "data/raw",
        "data/processed",
        "outputs",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    # Update paths config for Colab
    print("\n6. Configuring paths for Colab...")
    print(f"✓ Project root: {project_root}")
    print(f"✓ Outputs will be saved to: {project_root}/outputs")
    if drive_mounted:
        print("✓ Files will persist on Google Drive")
    else:
        print("⚠ WARNING: Files will be lost when Colab session ends!")
        print("  Consider mounting Drive for persistent storage")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set HF_TOKEN: os.environ['HF_TOKEN'] = 'your_token'")
    print("2. Run: python scripts/prepare_data.py")
    print("3. Run: python training/train.py --model_size 1B --output_dir ./outputs")
    print("\n" + "=" * 60)
    
    return project_root

if __name__ == "__main__":
    setup_colab()
