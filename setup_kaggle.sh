#!/bin/bash
# Setup Kaggle API for maximum dataset downloads

echo "=========================================="
echo "KAGGLE API SETUP"
echo "=========================================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
else
    echo "✓ Kaggle CLI already installed"
fi

echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Get your Kaggle API key:"
echo "   - Go to: https://www.kaggle.com/settings"
echo "   - Scroll to 'API' section"
echo "   - Click 'Create New API Token'"
echo "   - This downloads kaggle.json"
echo ""
echo "2. Install the API key:"
echo "   mkdir -p ~/.kaggle"
echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
echo "   chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "3. Test the API:"
echo "   kaggle datasets list -s diabetes"
echo ""
echo "4. Download all datasets:"
echo "   python3 download_real_datasets.py"
echo ""
echo "This will download 400K+ real patient records!"
echo ""
