#!/bin/bash
# Download NHANES 2017-2018 Data
# Public dataset - no approval needed

set -e  # Exit on error

echo "================================================"
echo "Downloading NHANES 2017-2018 Data"
echo "================================================"

# Create directory
mkdir -p data/nhanes/2017-2018
cd data/nhanes/2017-2018

BASE_URL="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018"

echo ""
echo "Downloading core data files..."
echo ""

# Demographics
echo "[1/11] Demographics..."
wget -q --show-progress "${BASE_URL}/DEMO_J.XPT"

# Laboratory - Glucose
echo "[2/11] Glucose..."
wget -q --show-progress "${BASE_URL}/GLU_J.XPT"

# Laboratory - Glycohemoglobin (HbA1c)
echo "[3/11] HbA1c..."
wget -q --show-progress "${BASE_URL}/GHB_J.XPT"

# Laboratory - Biochemistry Profile
echo "[4/11] Biochemistry (ALT, AST, Creatinine)..."
wget -q --show-progress "${BASE_URL}/BIOPRO_J.XPT"

# Laboratory - Cholesterol
echo "[5/11] Cholesterol (LDL, HDL, Triglycerides)..."
wget -q --show-progress "${BASE_URL}/TRIGLY_J.XPT"

# Laboratory - CRP
echo "[6/11] C-Reactive Protein..."
wget -q --show-progress "${BASE_URL}/CRP_J.XPT"

# Body Measures
echo "[7/11] Body Measures (BMI, Weight, Height)..."
wget -q --show-progress "${BASE_URL}/BMX_J.XPT"

# Blood Pressure
echo "[8/11] Blood Pressure..."
wget -q --show-progress "${BASE_URL}/BPX_J.XPT"

# Physical Activity Questionnaire
echo "[9/11] Physical Activity..."
wget -q --show-progress "${BASE_URL}/PAQ_J.XPT"

# Smoking Questionnaire
echo "[10/11] Smoking..."
wget -q --show-progress "${BASE_URL}/SMQ_J.XPT"

# Alcohol Questionnaire
echo "[11/11] Alcohol..."
wget -q --show-progress "${BASE_URL}/ALQ_J.XPT"

echo ""
echo "================================================"
echo "Download Complete!"
echo "================================================"
echo ""
echo "Files downloaded:"
ls -lh *.XPT
echo ""
echo "Total size:"
du -sh .
echo ""
echo "Next steps:"
echo "1. Test the loader: python3 examples/test_nhanes_loader.py"
echo "2. Process data: python3 examples/process_nhanes_data.py"
echo "3. Train model: python3 train_hybrid_model.py --data nhanes"
echo ""
