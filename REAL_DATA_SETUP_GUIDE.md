# 🔬 Real Medical Data Setup Guide

**How to Access and Use MIMIC-IV and NHANES Data**

---

## 📊 **Option 1: MIMIC-IV (ICU Clinical Data)** - HIGHEST QUALITY

### **What is MIMIC-IV?**
- **300,000+ ICU patient admissions** from Beth Israel Deaconess Medical Center
- **Comprehensive data:** Labs, vitals, medications, diagnoses, procedures
- **Time-series data:** Longitudinal measurements over hospital stay
- **Gold standard** for healthcare AI research

### **Step 1: Get Access (FREE but requires training)**

1. **Create PhysioNet Account**
   - Go to: https://physionet.org/register/
   - Create free account

2. **Complete CITI Training** (~4 hours, FREE)
   - Go to: https://physionet.org/about/citi-course/
   - Complete "Data or Specimens Only Research" course
   - **IMPORTANT:** Choose "Massachusetts Institute of Technology Affiliates"
   - Download completion certificate

3. **Sign Data Use Agreement**
   - Go to: https://physionet.org/content/mimiciv/2.2/
   - Click "Request Access"
   - Upload CITI certificate
   - Sign data use agreement
   - **Approval takes 1-3 days**

### **Step 2: Download MIMIC-IV**

Once approved:

```bash
# Install wget if needed
sudo apt-get install wget

# Create directory
mkdir -p /home/tc115/Yue/Patient_Digital_Twin_Systems/data/mimic-iv

# Download MIMIC-IV (requires PhysioNet credentials)
cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/mimic-iv

# Option A: Download via wget (after approval)
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/2.2/

# Option B: Download specific modules only (faster)
# Hospital data (demographics, labs, diagnoses)
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/2.2/hosp/

# ICU data (vitals, medications)
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/2.2/icu/
```

**Size:** ~50 GB compressed, ~200 GB uncompressed

### **Step 3: Use MIMIC-IV Loader**

```python
from data_integration.mimic_loader import MIMICLoader

# Initialize loader
loader = MIMICLoader('/home/tc115/Yue/Patient_Digital_Twin_Systems/data/mimic-iv')

# Get a cohort of patients
cohort = loader.get_cohort(
    min_age=40,
    max_age=70,
    has_labs=True,
    limit=1000
)

print(f"Found {len(cohort)} patients")

# Extract features for first patient
if cohort:
    patient_id = cohort[0]
    features = loader.extract_patient_features(patient_id)
    
    print(f"\nPatient {patient_id}:")
    print(f"  Age: {features.get('age')}")
    print(f"  Sex: {features.get('sex')}")
    print(f"  Glucose: {features.get('glucose')} mg/dL")
    print(f"  HbA1c: {features.get('hba1c')}%")
    print(f"  BP: {features.get('systolic_bp')}/{features.get('diastolic_bp')}")
```

---

## 📈 **Option 2: NHANES (Population Health Data)** - EASIEST ACCESS

### **What is NHANES?**
- **National Health and Nutrition Examination Survey** (CDC)
- **50,000+ participants** per cycle (every 2 years)
- **Publicly available** - no approval needed
- **Comprehensive:** Labs, body measures, lifestyle, diet, disease outcomes

### **Step 1: Download NHANES Data**

**No approval needed - publicly available!**

```bash
# Create directory
mkdir -p /home/tc115/Yue/Patient_Digital_Twin_Systems/data/nhanes/2017-2018

cd /home/tc115/Yue/Patient_Digital_Twin_Systems/data/nhanes/2017-2018

# Download key files (SAS transport format .XPT)

# Demographics
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT

# Laboratory - Glucose
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT

# Laboratory - Glycohemoglobin (HbA1c)
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT

# Laboratory - Standard Biochemistry Profile (ALT, AST, Creatinine)
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.XPT

# Laboratory - Cholesterol (LDL, HDL, Triglycerides)
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TRIGLY_J.XPT

# Laboratory - CRP
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/CRP_J.XPT

# Body Measures (BMI, weight, height, waist)
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT

# Blood Pressure
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT

# Questionnaires
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT  # Physical activity
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT  # Smoking
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.XPT  # Alcohol
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SLQ_J.XPT  # Sleep
wget https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DBQ_J.XPT  # Diet
```

**Size:** ~500 MB total

### **Step 2: Install Required Library**

```bash
pip install pandas numpy scipy
```

### **Step 3: Use NHANES Loader**

```python
from data_integration.nhanes_loader import NHANESLoader

# Initialize loader
loader = NHANESLoader(
    '/home/tc115/Yue/Patient_Digital_Twin_Systems/data/nhanes',
    cycle='2017-2018'
)

# Get a cohort
cohort = loader.get_cohort(
    min_age=40,
    max_age=70,
    has_labs=True,
    limit=1000
)

print(f"Found {len(cohort)} patients")

# Extract features for first patient
if cohort:
    seqn = cohort[0]
    features = loader.extract_patient_features(seqn)
    
    print(f"\nPatient {seqn}:")
    print(f"  Age: {features.get('age')}")
    print(f"  Sex: {features.get('sex')}")
    print(f"  BMI: {features.get('bmi')}")
    print(f"  Glucose: {features.get('fasting_glucose')} mg/dL")
    print(f"  HbA1c: {features.get('hba1c')}%")
    print(f"  BP: {features.get('systolic_bp')}/{features.get('diastolic_bp')}")
    print(f"  Activity: {features.get('physical_activity')}")
```

---

## 🔄 **Step 4: Harmonize and Extract Features**

Once you have either MIMIC-IV or NHANES data:

```python
from data_integration.data_harmonizer import DataHarmonizer
from data_integration.feature_extractor import FeatureExtractor

# Initialize
harmonizer = DataHarmonizer()
extractor = FeatureExtractor()

# Load data from MIMIC-IV or NHANES
# (using NHANES as example)
loader = NHANESLoader('./data/nhanes', cycle='2017-2018')
cohort = loader.get_cohort(min_age=40, max_age=70, limit=5000)

# Process all patients
all_features = []
for seqn in cohort:
    # Extract raw features
    raw_data = loader.extract_patient_features(seqn)
    
    # Harmonize to standard format
    harmonized = harmonizer.harmonize(raw_data, source='nhanes')
    
    # Validate
    is_valid, errors = harmonizer.validate(harmonized)
    if not is_valid:
        print(f"Skipping patient {seqn}: {errors}")
        continue
    
    # Extract ML features
    ml_features = extractor.extract_all_features(harmonized)
    
    all_features.append({
        'patient_id': harmonized['patient_id'],
        'features': ml_features,
        'labels': {
            'has_diabetes': harmonized['has_diabetes'],
            'has_hypertension': harmonized['has_hypertension'],
            'has_ckd': harmonized['has_ckd']
        }
    })

print(f"Processed {len(all_features)} patients")

# Save for training
import pickle
with open('./data/processed_features.pkl', 'wb') as f:
    pickle.dump(all_features, f)
```

---

## 🚀 **Quick Start: Download NHANES Now**

**Fastest way to get real data (no approval needed):**

```bash
# Run this script to download NHANES 2017-2018
cd /home/tc115/Yue/Patient_Digital_Twin_Systems

# Create download script
cat > download_nhanes.sh << 'EOF'
#!/bin/bash
mkdir -p data/nhanes/2017-2018
cd data/nhanes/2017-2018

echo "Downloading NHANES 2017-2018 data..."

# Core files
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TRIGLY_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/CRP_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT
wget -q --show-progress https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.XPT

echo "Download complete!"
ls -lh
EOF

chmod +x download_nhanes.sh
./download_nhanes.sh
```

---

## 📊 **Data Comparison**

| Feature | MIMIC-IV | NHANES | Synthetic |
|---------|----------|--------|-----------|
| **Size** | 300,000+ patients | 50,000+ people | Unlimited |
| **Access** | Requires approval (1-3 days) | Public (immediate) | Immediate |
| **Cost** | Free | Free | Free |
| **Quality** | ⭐⭐⭐⭐⭐ ICU data | ⭐⭐⭐⭐ Population | ⭐⭐⭐ Realistic |
| **Time-series** | ✅ Yes (hourly) | ❌ Single timepoint | ✅ Can generate |
| **Completeness** | ⭐⭐⭐ (ICU focus) | ⭐⭐⭐⭐⭐ Comprehensive | ⭐⭐⭐⭐⭐ Complete |
| **Best for** | Critical care, acute | Population health | Initial development |

---

## 🎯 **Recommended Approach**

### **Phase 1: Start with NHANES (Today)**
1. ✅ Download NHANES (30 minutes)
2. ✅ Test loaders and harmonizer
3. ✅ Extract features for 5,000 patients
4. ✅ Train initial model

### **Phase 2: Add MIMIC-IV (This Week)**
1. ⏳ Complete CITI training (4 hours)
2. ⏳ Request MIMIC-IV access
3. ⏳ Wait for approval (1-3 days)
4. ⏳ Download MIMIC-IV data
5. ⏳ Fine-tune model on ICU data

### **Phase 3: Combine Both (Next Week)**
1. ⏳ Train on NHANES (population baseline)
2. ⏳ Fine-tune on MIMIC-IV (clinical refinement)
3. ⏳ Validate on held-out test sets
4. ⏳ Publish results

---

## ✅ **Next Steps**

**Choose one:**

### **Option A: Quick Start with NHANES (Recommended)**
```bash
cd /home/tc115/Yue/Patient_Digital_Twin_Systems
./download_nhanes.sh  # Downloads in 30 min
python3 examples/load_nhanes_data.py  # Test loader
python3 train_hybrid_model.py  # Train model
```

### **Option B: Wait for MIMIC-IV (Higher Quality)**
1. Go to https://physionet.org/register/
2. Complete CITI training
3. Request MIMIC-IV access
4. Download when approved
5. Train model

### **Option C: Use Both (Best)**
1. Start with NHANES today
2. Apply for MIMIC-IV in parallel
3. Train on NHANES first
4. Add MIMIC-IV when approved

---

**Which option would you like to pursue?**

I can help you:
1. Download NHANES right now (30 minutes)
2. Guide you through MIMIC-IV application
3. Create training scripts for either dataset
