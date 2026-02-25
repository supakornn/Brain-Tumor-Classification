#!/bin/bash

# Brain Tumor MRI Dataset Downloader
# Complete script to download and setup the dataset

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Brain Tumor MRI Dataset Downloader  ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Step 1: Check UV
echo -e "${CYAN}[1/5] Checking UV...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ UV not found${NC}"
    echo ""
    echo "Please install UV first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✓ UV is installed${NC}"
echo ""

# Step 2: Setup virtual environment
echo -e "${CYAN}[2/5] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    uv sync
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Step 3: Install Kaggle CLI
echo -e "${CYAN}[3/5] Installing Kaggle CLI...${NC}"
if ! .venv/bin/python -c "import kaggle" &> /dev/null; then
    uv add kaggle
    echo -e "${GREEN}✓ Kaggle CLI installed${NC}"
else
    echo -e "${GREEN}✓ Kaggle CLI already installed${NC}"
fi
echo ""

# Step 4: Check/Setup Kaggle credentials
echo -e "${CYAN}[4/5] Checking Kaggle credentials...${NC}"
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${YELLOW}⚠ Kaggle credentials not found${NC}"
    echo ""
    echo "To download the dataset, you need Kaggle API credentials."
    echo ""
    echo -e "${CYAN}Setup Instructions:${NC}"
    echo "  1. Go to: https://www.kaggle.com/settings"
    echo "  2. Scroll to 'API' section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Enter your credentials below"
    echo ""
    
    read -p "Enter your Kaggle username: " kaggle_username
    read -p "Enter your Kaggle API key: " kaggle_key
    
    if [ -z "$kaggle_username" ] || [ -z "$kaggle_key" ]; then
        echo -e "${RED}✗ Username and API key are required${NC}"
        exit 1
    fi
    
    # Create credentials file
    mkdir -p ~/.kaggle
    cat > ~/.kaggle/kaggle.json << EOF
{"username":"$kaggle_username","key":"$kaggle_key"}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    
    echo -e "${GREEN}✓ Kaggle credentials saved${NC}"
else
    echo -e "${GREEN}✓ Kaggle credentials found${NC}"
fi
echo ""

# Step 5: Download dataset
echo -e "${CYAN}[5/5] Downloading dataset...${NC}"
echo ""
echo "Dataset: Brain Tumor MRI Dataset"
echo "Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
echo "Size: ~95 MB"
echo ""

if [ -d "data/Training" ] && [ -d "data/Testing" ]; then
    echo -e "${YELLOW}⚠ Dataset already exists${NC}"
    read -p "Do you want to re-download? (y/N): " redownload
    if [[ ! $redownload =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        echo ""
        echo -e "${GREEN}✓ Dataset ready${NC}"
        exit 0
    fi
    echo ""
    echo "Removing old data..."
    rm -rf data/
fi

echo "Downloading... (this may take a few minutes)"
uv run kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data --unzip

echo ""
echo -e "${GREEN}✓ Download complete!${NC}"
echo ""

# Verify and show statistics
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Dataset Statistics                   ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [ -d "data/Training" ]; then
    echo "Training Set:"
    for folder in data/Training/*/; do
        if [ -d "$folder" ]; then
            count=$(find "$folder" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
            folder_name=$(basename "$folder")
            printf "  %-20s: %s images\n" "$folder_name" "$count"
        fi
    done
    echo ""
fi

if [ -d "data/Testing" ]; then
    echo "Testing Set:"
    for folder in data/Testing/*/; do
        if [ -d "$folder" ]; then
            count=$(find "$folder" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
            folder_name=$(basename "$folder")
            printf "  %-20s: %s images\n" "$folder_name" "$count"
        fi
    done
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete! 🎉                   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Explore data:  uv run python scripts/data_exploration.py"
echo "  2. Train model:   uv run python scripts/train_models.py"
echo "  3. Make prediction: uv run python scripts/predict_single_image.py <image_path>"
echo ""
