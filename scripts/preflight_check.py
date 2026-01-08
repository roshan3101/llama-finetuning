"""
Pre-flight check script.

Validates all prerequisites before running the fine-tuning pipeline.
Checks Python version, dependencies, Hugging Face access, datasets, GPU, etc.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env file from {env_path}")
except ImportError:
    # If python-dotenv not installed, try manual loading
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    except Exception:
        pass

# Color codes for terminal output (with fallback for Windows)
try:
    class Colors:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        END = '\033[0m'
        BOLD = '\033[1m'
    CHECK = '✓'
    CROSS = '✗'
    WARN = '⚠'
except:
    # Fallback for systems without color support
    class Colors:
        GREEN = ''
        RED = ''
        YELLOW = ''
        BLUE = ''
        END = ''
        BOLD = ''
    CHECK = '[OK]'
    CROSS = '[X]'
    WARN = '[!]'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    try:
        print(f"{Colors.GREEN}{CHECK}{Colors.END} {text}")
    except UnicodeEncodeError:
        print(f"[OK] {text}")

def print_error(text: str):
    """Print error message."""
    try:
        print(f"{Colors.RED}{CROSS}{Colors.END} {text}")
    except UnicodeEncodeError:
        print(f"[X] {text}")

def print_warning(text: str):
    """Print warning message."""
    try:
        print(f"{Colors.YELLOW}{WARN}{Colors.END} {text}")
    except UnicodeEncodeError:
        print(f"[!] {text}")

def print_info(text: str):
    """Print info message."""
    print(f"  {text}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required packages are installed."""
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
        "accelerate",
        "huggingface_hub",
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "huggingface_hub":
                __import__("huggingface_hub")
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
        except Exception:
            # Some packages might have import errors even if installed
            # (e.g., bitsandbytes on CPU-only systems)
            pass
    
    return len(missing) == 0, missing

def check_hf_token() -> Tuple[bool, str]:
    """Check if Hugging Face token is set."""
    # Try multiple environment variable names
    token = (
        os.getenv("HF_TOKEN") or 
        os.getenv("HUGGINGFACE_TOKEN") or 
        os.getenv("HF_API_TOKEN") or
        os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    
    # Also check .env file directly if not in environment
    if not token:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    
    if token:
        # Check if it's a placeholder
        if "your" in token.lower() or "token" in token.lower() or len(token) < 10:
            return False, "Token appears to be a placeholder - replace with actual token"
        # Check token format (HF tokens usually start with 'hf_' and are 37+ chars)
        if not token.startswith("hf_") and len(token) < 20:
            return False, f"Token format may be incorrect (should start with 'hf_' and be ~37 chars, got {len(token)} chars)"
        return True, f"Token found (length: {len(token)})"
    return False, "Token not found in environment variables or .env file"

def check_hf_login() -> Tuple[bool, str]:
    """Check if logged in to Hugging Face."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        if user_info:
            return True, f"Logged in as: {user_info.get('name', 'Unknown')}"
    except ImportError:
        return False, "huggingface_hub not installed"
    except Exception as e:
        return False, f"Not logged in: {str(e)}"
    return False, "Not logged in"

def check_model_access(model_name: str) -> Tuple[bool, str]:
    """Check if model is accessible."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        return True, f"Access granted to {model_name}"
    except ImportError:
        return False, "transformers package not installed"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, f"Access denied - request access at https://huggingface.co/{model_name}"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return False, f"Model not found: {model_name}"
        else:
            return False, f"Error: {error_msg[:100]}"

def check_dataset_access(dataset_name: str) -> Tuple[bool, str]:
    """Check if dataset is accessible."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split="train[:1]", trust_remote_code=True)
        return True, f"Accessible: {dataset_name} ({len(dataset)} examples in sample)"
    except ImportError:
        return False, "datasets package not installed"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, f"Access denied - may need to request access"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return False, f"Dataset not found: {dataset_name}"
        else:
            return False, f"Error: {error_msg[:100]}"

def check_gpu() -> Tuple[bool, Dict]:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, {
                "available": True,
                "name": gpu_name,
                "memory_gb": round(gpu_memory, 2)
            }
        else:
            return False, {"available": False, "message": "CUDA not available"}
    except ImportError:
        return False, {"available": False, "message": "torch package not installed"}
    except Exception as e:
        return False, {"available": False, "message": f"Error checking GPU: {str(e)}"}

def check_directories() -> Tuple[bool, List[str]]:
    """Check if required directories exist or can be created."""
    required_dirs = [
        "data/raw",
        "data/processed",
        "outputs",
        "logs",
    ]
    
    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                missing.append(f"{dir_path} (cannot create: {str(e)})")
    
    return len(missing) == 0, missing

def check_data_files() -> Tuple[bool, Dict]:
    """Check if data files exist."""
    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/val.jsonl")
    
    results = {
        "train_exists": train_path.exists(),
        "val_exists": val_path.exists(),
        "train_size": train_path.stat().st_size if train_path.exists() else 0,
        "val_size": val_path.stat().st_size if val_path.exists() else 0,
    }
    
    if results["train_exists"] and results["val_exists"]:
        return True, results
    return False, results

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space (rough estimate)."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb < 10:
            return False, f"Low disk space: {free_gb:.2f} GB free (need at least 10 GB)"
        return True, f"Disk space OK: {free_gb:.2f} GB free"
    except Exception:
        return True, "Could not check disk space"

def run_all_checks(model_size: str = "1B") -> Dict:
    """Run all pre-flight checks."""
    results = {
        "passed": [],
        "failed": [],
        "warnings": [],
        "ready": True,
    }
    
    print_header("Pre-Flight Check for Fine-Tuning Pipeline")
    
    # 1. Python version
    print(f"{Colors.BOLD}1. Python Version{Colors.END}")
    passed, msg = check_python_version()
    if passed:
        print_success(msg)
        results["passed"].append(("Python Version", msg))
    else:
        print_error(msg)
        results["failed"].append(("Python Version", msg))
        results["ready"] = False
    
    # 2. Dependencies
    print(f"\n{Colors.BOLD}2. Dependencies{Colors.END}")
    passed, missing = check_dependencies()
    if passed:
        print_success("All required packages installed")
        results["passed"].append(("Dependencies", "All installed"))
    else:
        print_error(f"Missing packages: {', '.join(missing)}")
        print_info("Run: pip install -r requirements.txt")
        results["failed"].append(("Dependencies", f"Missing: {', '.join(missing)}"))
        results["ready"] = False
    
    # 3. Hugging Face Token
    print(f"\n{Colors.BOLD}3. Hugging Face Token{Colors.END}")
    passed, msg = check_hf_token()
    if passed:
        print_success(msg)
        results["passed"].append(("HF Token", "Set"))
    else:
        print_error(msg)
        print_info("Set HF_TOKEN environment variable or add to .env file")
        results["failed"].append(("HF Token", msg))
        results["ready"] = False
    
    # 4. Hugging Face Login
    print(f"\n{Colors.BOLD}4. Hugging Face Login{Colors.END}")
    passed, msg = check_hf_login()
    if passed:
        print_success(msg)
        results["passed"].append(("HF Login", msg))
    else:
        print_warning(msg)
        print_info("Run: huggingface-cli login")
        results["warnings"].append(("HF Login", msg))
    
    # 5. Model Access
    print(f"\n{Colors.BOLD}5. Model Access{Colors.END}")
    model_configs = {
        "1B": "meta-llama/Llama-3.2-1B-Instruct",
        "8B": "meta-llama/Llama-3.1-8B-Instruct",
        "70B": "meta-llama/Llama-3.1-70B-Instruct",
    }
    
    model_name = model_configs.get(model_size, model_configs["1B"])
    print_info(f"Checking access to: {model_name}")
    passed, msg = check_model_access(model_name)
    if passed:
        print_success(msg)
        results["passed"].append(("Model Access", f"{model_size} model accessible"))
    else:
        # Check if it's a gated repo (access requested but pending)
        if "gated" in msg.lower() or "access" in msg.lower():
            print_warning(f"Model is gated: {msg}")
            print_info(f"Request access at: https://huggingface.co/{model_name}")
            print_info("If you've already requested access, wait for approval (usually instant)")
            print_info("You can proceed with data preparation while waiting")
            results["warnings"].append(("Model Access", "Access requested, waiting for approval"))
        else:
            print_error(msg)
            results["failed"].append(("Model Access", msg))
            results["ready"] = False
    
    # 6. Dataset Access
    print(f"\n{Colors.BOLD}6. Dataset Access{Colors.END}")
    datasets = [
        "facebook/empathetic_dialogues",
        "google-research-datasets/go_emotions",
        "Amod/mental_health_counseling_conversations",
        "Pradeep016/career-guidance-qa-dataset",
        "ElenaSenger/Karrierewege",
    ]
    
    dataset_results = []
    for ds_name in datasets:
        passed, msg = check_dataset_access(ds_name)
        if passed:
            print_success(f"{ds_name}")
            dataset_results.append(True)
        else:
            print_warning(f"{ds_name}: {msg}")
            dataset_results.append(False)
    
    if all(dataset_results):
        results["passed"].append(("Datasets", "All accessible"))
    else:
        results["warnings"].append(("Datasets", "Some datasets may not be accessible"))
    
    # 7. GPU Check
    print(f"\n{Colors.BOLD}7. GPU Availability{Colors.END}")
    passed, gpu_info = check_gpu()
    if passed:
        print_success(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
        results["passed"].append(("GPU", f"{gpu_info['name']} ({gpu_info['memory_gb']} GB)"))
        
        # Check if GPU memory is sufficient
        if model_size == "1B" and gpu_info["memory_gb"] < 4:
            print_warning("1B model recommended: 4GB+ VRAM")
        elif model_size == "8B" and gpu_info["memory_gb"] < 16:
            print_warning("8B model recommended: 16GB+ VRAM")
        elif model_size == "70B" and gpu_info["memory_gb"] < 40:
            print_warning("70B model recommended: 40GB+ VRAM")
    else:
        print_warning(f"GPU: {gpu_info.get('message', 'Not available')}")
        print_info("Training will use CPU (very slow)")
        results["warnings"].append(("GPU", gpu_info.get("message", "Not available")))
    
    # 8. Directories
    print(f"\n{Colors.BOLD}8. Directory Structure{Colors.END}")
    passed, missing = check_directories()
    if passed:
        print_success("All required directories exist")
        results["passed"].append(("Directories", "OK"))
    else:
        print_error(f"Cannot create directories: {', '.join(missing)}")
        results["failed"].append(("Directories", f"Missing: {', '.join(missing)}"))
        results["ready"] = False
    
    # 9. Data Files
    print(f"\n{Colors.BOLD}9. Data Files{Colors.END}")
    passed, data_info = check_data_files()
    if passed:
        train_size_mb = data_info["train_size"] / (1024 * 1024)
        val_size_mb = data_info["val_size"] / (1024 * 1024)
        print_success(f"Training data: {train_size_mb:.2f} MB")
        print_success(f"Validation data: {val_size_mb:.2f} MB")
        results["passed"].append(("Data Files", "Exist"))
    else:
        print_warning("Data files not found")
        print_info("Run: python scripts/prepare_data.py")
        results["warnings"].append(("Data Files", "Not prepared yet"))
    
    # 10. Disk Space
    print(f"\n{Colors.BOLD}10. Disk Space{Colors.END}")
    passed, msg = check_disk_space()
    if passed:
        print_success(msg)
        results["passed"].append(("Disk Space", msg))
    else:
        print_warning(msg)
        results["warnings"].append(("Disk Space", msg))
    
    # Summary
    print_header("Summary")
    
    print(f"{Colors.BOLD}Passed:{Colors.END} {len(results['passed'])}")
    for check, msg in results["passed"]:
        try:
            print(f"  {Colors.GREEN}{CHECK}{Colors.END} {check}")
        except UnicodeEncodeError:
            print(f"  [OK] {check}")
    
    if results["warnings"]:
        print(f"\n{Colors.BOLD}Warnings:{Colors.END} {len(results['warnings'])}")
        for check, msg in results["warnings"]:
            try:
                print(f"  {Colors.YELLOW}{WARN}{Colors.END} {check}: {msg}")
            except UnicodeEncodeError:
                print(f"  [!] {check}: {msg}")
    
    if results["failed"]:
        print(f"\n{Colors.BOLD}Failed:{Colors.END} {len(results['failed'])}")
        for check, msg in results["failed"]:
            try:
                print(f"  {Colors.RED}{CROSS}{Colors.END} {check}: {msg}")
            except UnicodeEncodeError:
                print(f"  [X] {check}: {msg}")
    
    # Final verdict
    print_header("Verdict")
    
    if results["ready"]:
        try:
            print(f"{Colors.GREEN}{Colors.BOLD}{CHECK} READY TO START{Colors.END}")
        except UnicodeEncodeError:
            print(f"[OK] READY TO START")
        if results["warnings"]:
            print(f"\n{Colors.YELLOW}Note: There are {len(results['warnings'])} warnings, but you can proceed.{Colors.END}")
        return results
    else:
        try:
            print(f"{Colors.RED}{Colors.BOLD}{CROSS} NOT READY{Colors.END}")
        except UnicodeEncodeError:
            print(f"[X] NOT READY")
        print(f"\nPlease fix the {len(results['failed'])} critical issue(s) above before starting.")
        return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-flight check for fine-tuning pipeline")
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        choices=["1B", "8B", "70B"],
        help="Model size to check access for"
    )
    
    args = parser.parse_args()
    
    results = run_all_checks(model_size=args.model_size)
    
    # Exit with appropriate code
    sys.exit(0 if results["ready"] else 1)

if __name__ == "__main__":
    main()

