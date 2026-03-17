# Real Data Sources - Practical Guide

## 🎯 How to Get Real Public Health Datasets

The automated acquisition found 2,721 datasets but couldn't download them automatically. Here's how to get real data manually.

---

## 📊 Best Public Health Datasets

### **1. Kaggle (Easiest - Direct Downloads)**

**Setup:**
```bash
# Install Kaggle CLI
pip install kaggle

# Get API key from https://www.kaggle.com/settings
# Download kaggle.json and place it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Download Datasets:**
```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/real/raw

# Diabetes datasets
kaggle datasets download -d uciml/pima-indians-diabetes-database
kaggle datasets download -d mathchi/diabetes-data-set
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset

# Cardiovascular datasets
kaggle datasets download -d sulianova/cardiovascular-disease-dataset
kaggle datasets download -d johnsmith88/heart-disease-dataset

# General health datasets
kaggle datasets download -d cdc/behavioral-risk-factor-surveillance-system
kaggle datasets download -d cdc/national-health-and-nutrition-examination-survey

# Unzip
unzip '*.zip'
```

**Recommended Kaggle Datasets:**
- **Pima Indians Diabetes** - 768 patients, classic dataset
- **Cardiovascular Disease** - 70,000 patients
- **Heart Disease UCI** - 303 patients, 14 features
- **BRFSS** - 400,000+ patients, comprehensive health survey

---

### **2. UCI Machine Learning Repository**

**Direct Downloads:**
```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/real/raw

# Heart Disease
wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
mv processed.cleveland.data heart_disease_uci.csv

# Diabetes
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
unzip dataset_diabetes.zip

# Chronic Kidney Disease
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.arff
```

**Available Datasets:**
- Heart Disease (Cleveland) - 303 patients
- Diabetes 130-US hospitals - 100,000 patients
- Chronic Kidney Disease - 400 patients
- Thyroid Disease - 7,200 patients

---

### **3. PhysioNet (Medical Research)**

**Requires Free Registration:** https://physionet.org/

**After Registration:**
```bash
# Install wfdb package
pip install wfdb

# Download MIMIC-III Demo (no credentialing required)
wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/
```

**Available Datasets:**
- **MIMIC-III Demo** - 100 ICU patients (free)
- **MIMIC-III Full** - 40,000 ICU patients (requires credentialing)
- **eICU** - 200,000 ICU admissions (requires credentialing)
- **PTB-XL** - 21,000 ECG recordings

---

### **4. Data.gov (US Government)**

**Direct Downloads:**
```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/real/raw

# Medicare data
wget https://data.cms.gov/provider-data/sites/default/files/resources/...

# CDC NHANES
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT
```

**Available:**
- **NHANES** - 10,000+ patients per cycle, comprehensive health data
- **Medicare Claims** - Millions of records
- **CDC Behavioral Risk** - 400,000+ patients annually

---

### **5. Zenodo (Research Datasets)**

**Manual Download:**
1. Go to https://zenodo.org/
2. Search: "diabetes patient data" or "cardiovascular health"
3. Click dataset → Download files

**Example Datasets:**
- Search: "clinical trial diabetes"
- Search: "patient health records"
- Search: "cardiovascular disease dataset"

---

## 🚀 Quick Start Script

I'll create an automated downloader for the easiest sources:

```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems

# Run automated download
python3 download_real_datasets.py
```

This will:
1. Download from Kaggle (if API key configured)
2. Download from UCI repository
3. Download from Data.gov
4. Organize all files
5. Generate summary report

---

## 📊 Expected Results

**After running downloads:**
- **Kaggle:** 5-10 datasets, 100K-500K patients
- **UCI:** 3-5 datasets, 10K-100K patients
- **Data.gov:** 2-3 datasets, 100K+ patients
- **Total:** 200K-700K real patients

**Storage:** 500 MB - 2 GB

---

## 🔧 Setup Kaggle API (Recommended)

**Step 1:** Get API key
```
1. Go to https://www.kaggle.com/
2. Sign in (or create account)
3. Click your profile → Settings
4. Scroll to "API" section
5. Click "Create New API Token"
6. Download kaggle.json
```

**Step 2:** Install
```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 3:** Test
```bash
kaggle datasets list -s diabetes
```

---

## 📝 Manual Download Instructions

If automated download fails, here's the manual process:

### **For Kaggle:**
1. Go to dataset page (e.g., https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. Click "Download" button
3. Move to `data/real/raw/`
4. Unzip

### **For UCI:**
1. Go to https://archive.ics.uci.edu/ml/datasets.php
2. Search for dataset
3. Click "Data Folder"
4. Download files
5. Move to `data/real/raw/`

### **For PhysioNet:**
1. Register at https://physionet.org/
2. Browse datasets
3. Click "Files" tab
4. Download
5. Move to `data/real/raw/`

---

## 🎯 Recommended Workflow

**Priority 1: Kaggle (Easiest)**
```bash
# Setup Kaggle API
pip install kaggle
# Configure API key (see above)

# Download top datasets
kaggle datasets download -d uciml/pima-indians-diabetes-database
kaggle datasets download -d sulianova/cardiovascular-disease-dataset
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset
```

**Priority 2: UCI Repository**
```bash
# Direct wget downloads (no registration)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
```

**Priority 3: Data.gov**
```bash
# NHANES data (large, comprehensive)
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
```

---

## 💡 Why These Sources?

**Kaggle:**
- ✅ Easy API access
- ✅ Pre-cleaned datasets
- ✅ Large community
- ✅ Direct downloads

**UCI:**
- ✅ Classic ML datasets
- ✅ Well-documented
- ✅ No registration
- ✅ Direct downloads

**PhysioNet:**
- ✅ Medical-grade data
- ✅ Large sample sizes
- ⚠️ Requires registration
- ⚠️ Some require credentialing

**Data.gov:**
- ✅ Government data
- ✅ Very large datasets
- ✅ Free access
- ⚠️ Complex formats

---

## 🎯 Next Steps

1. **Setup Kaggle API** (5 minutes)
2. **Run download script** (I'll create this)
3. **Train models on real data**
4. **Compare with synthetic models**

The automated script will handle everything once Kaggle API is configured!
