#!/bin/bash

# Arctic Ice Monitoring - UV Setup Script
# This script sets up the development environment using UV

set -e  # Exit on error

echo "=================================="
echo "Arctic Ice Monitoring - UV Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV is not installed. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo -e "${GREEN}✓ UV installed successfully${NC}"
    echo ""
    echo -e "${YELLOW}Note: Add this to your ~/.bashrc or ~/.zshrc:${NC}"
    echo "export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    echo ""
else
    echo -e "${GREEN}✓ UV is already installed${NC}"
fi

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION (>= 3.11)${NC}"
else
    echo -e "${RED}✗ Python 3.11+ required. Current: $PYTHON_VERSION${NC}"
    exit 1
fi

# Ask user which components to install
echo ""
echo "Which components do you want to install?"
echo "1) Full installation (Backend + ML + Dev tools) - Recommended"
echo "2) Backend only (FastAPI, Database, API)"
echo "3) ML only (Training, Notebooks)"
echo "4) ML + Dev (Training, Notebooks, Jupyter)"
echo "5) Custom"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        INSTALL_ARGS=".[all]"
        COMPONENT="Full installation"
        ;;
    2)
        INSTALL_ARGS=".[backend]"
        COMPONENT="Backend only"
        ;;
    3)
        INSTALL_ARGS=".[ml]"
        COMPONENT="ML only"
        ;;
    4)
        INSTALL_ARGS=".[ml,dev]"
        COMPONENT="ML + Dev"
        ;;
    5)
        echo ""
        echo "Available components: backend, ml, dev"
        read -p "Enter components (comma-separated, e.g., 'backend,dev'): " custom
        INSTALL_ARGS=".[${custom}]"
        COMPONENT="Custom: $custom"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf .venv
fi

uv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "Installing dependencies: $COMPONENT"
echo "This may take 1-2 minutes..."
echo ""

uv pip install -e "$INSTALL_ARGS"

echo ""
echo -e "${GREEN}✓ Dependencies installed successfully${NC}"

# Show installed packages
echo ""
echo "Installed packages:"
uv pip list | head -20
TOTAL_PACKAGES=$(uv pip list | wc -l)
echo "... and $((TOTAL_PACKAGES - 20)) more packages"

# Final instructions
echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete! 🎉${NC}"
echo "=================================="
echo ""
echo "To activate the virtual environment in the future:"
echo -e "${YELLOW}source .venv/bin/activate${NC}"
echo ""

if [[ $choice == "1" || $choice == "2" || $choice == "5" && $custom == *"backend"* ]]; then
    echo "To run the backend:"
    echo "  cd backend"
    echo "  uvicorn app.main:app --reload"
    echo ""
fi

if [[ $choice == "1" || $choice == "3" || $choice == "4" ]]; then
    echo "To train ML models:"
    echo "  cd ml-models"
    echo "  python train_all.py"
    echo ""
fi

if [[ $choice == "1" || $choice == "4" ]]; then
    echo "To use Jupyter notebooks:"
    echo "  cd ml-models/notebooks"
    echo "  jupyter notebook"
    echo ""
fi

echo "For more information, see:"
echo "  - UV_SETUP.md"
echo "  - README.md"
echo "  - ml-models/ML_TRAINING_GUIDE.md"
echo ""
