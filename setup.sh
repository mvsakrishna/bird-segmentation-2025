#!/bin/bash
# =============================================================================
#  Bird Segmentation 2025 — One-Command Setup Script
#  Usage: bash setup.sh
# =============================================================================

set -e  # exit immediately on any error

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Bird Segmentation 2025 — Environment Setup           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ─────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "❌  Python 3.10+ is required. Current version: $PYTHON_VERSION"
    echo "    Install from https://www.python.org/downloads/"
    exit 1
fi
echo "✅  Python $PYTHON_VERSION detected."

# ── 2. Create virtual environment ───────────────────────────────────────────
if [ ! -d "birdseg_env" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv birdseg_env
fi

# Activate (works for bash/zsh on Linux/Mac)
source birdseg_env/bin/activate
echo "✅  Virtual environment activated."

# ── 3. Upgrade pip ──────────────────────────────────────────────────────────
echo "⬆️   Upgrading pip..."
pip install --upgrade pip --quiet

# ── 4. Install PyTorch (auto-detect CUDA) ───────────────────────────────────
echo ""
echo "🔍  Detecting GPU..."

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "✅  CUDA GPU already available via existing torch installation."
elif command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "✅  NVIDIA GPU detected. CUDA $CUDA_VERSION"
    echo "🔧  Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "⚠️   No GPU detected. Installing CPU-only PyTorch (slower inference)."
    pip install torch torchvision torchaudio --quiet
fi

# ── 5. Install project dependencies ─────────────────────────────────────────
echo ""
echo "📥  Installing project dependencies..."
pip install -r requirements.txt --quiet
echo "✅  Dependencies installed."

# ── 6. Install Grounding DINO ───────────────────────────────────────────────
echo ""
echo "📥  Installing Grounding DINO..."
pip install groundingdino-py --quiet

# Download Grounding DINO weights
mkdir -p weights
GDINO_WEIGHTS="weights/groundingdino_swinb_cogcoor.pth"
if [ ! -f "$GDINO_WEIGHTS" ]; then
    echo "⬇️   Downloading Grounding DINO weights (~660 MB)..."
    wget -q --show-progress \
        -O "$GDINO_WEIGHTS" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    echo "✅  Grounding DINO weights saved to $GDINO_WEIGHTS"
else
    echo "✅  Grounding DINO weights already present."
fi

# ── 7. Verify installation ──────────────────────────────────────────────────
echo ""
echo "🔎  Verifying installation..."
python3 - <<'EOF'
import sys

results = {}

try:
    import torch
    results["PyTorch"]   = torch.__version__
    results["CUDA"]      = str(torch.cuda.is_available())
except Exception as e:
    results["PyTorch"]   = f"FAIL: {e}"

try:
    import ultralytics
    results["Ultralytics"] = ultralytics.__version__
except Exception as e:
    results["Ultralytics"] = f"FAIL: {e}"

try:
    import groundingdino
    results["GroundingDINO"] = "OK"
except Exception as e:
    results["GroundingDINO"] = f"FAIL: {e}"

try:
    import cv2
    results["OpenCV"] = cv2.__version__
except Exception as e:
    results["OpenCV"] = f"FAIL: {e}"

try:
    import albumentations
    results["Albumentations"] = albumentations.__version__
except Exception as e:
    results["Albumentations"] = f"FAIL: {e}"

print("")
for k, v in results.items():
    status = "✅" if "FAIL" not in str(v) else "❌"
    print(f"  {status}  {k:<20} {v}")

failures = [k for k, v in results.items() if "FAIL" in str(v)]
if failures:
    print(f"\n❌  {len(failures)} package(s) failed to install. See above.")
    sys.exit(1)
else:
    print("\n✅  All packages verified successfully!")
EOF

# ── 8. Done ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  To activate the environment in future sessions:"
echo "    source birdseg_env/bin/activate"
echo ""
echo "  Quick start (zero-shot, no dataset needed):"
echo "    python bird_seg_2025.py --demo /path/to/any/bird/image.jpg"
echo ""
echo "  Full pipeline with CUB-200-2011:"
echo "    python bird_seg_2025.py --prepare-data --cub-root ./CUB_200_2011"
echo "    python bird_seg_2025.py --mode train"
echo "    python bird_seg_2025.py --mode eval"
echo ""
