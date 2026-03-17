# Real Data Training Pipeline - Quick Start

## 🎯 Overview

You now have **two parallel data systems**:

1. **Synthetic Data** (5M patients) - For development/testing
2. **Real Data** (Automated acquisition) - For production/validation

---

## 🚀 Current Status

### **Real Data Acquisition** ✅ RUNNING
```bash
# Check status
ps aux | grep run_daily_data_acquisition
```

**What's happening:**
- Searching Figshare, Zenodo, Data.gov, Kaggle
- Found 1,500+ datasets so far
- Downloading up to 10GB
- Validating and cleaning automatically

**Data location:** `data/real/`

---

## 📊 Training on Real Data

### **Step 1: Wait for Data Acquisition to Complete**

Check progress:
```bash
# View logs
tail -f logs/daily_acquisition.log

# Check downloaded datasets
ls -lh data/real/raw/

# Check lineage
cat data/real/lineage.json | jq '.statistics'
```

### **Step 2: Train Models on Real Data**

Once datasets are downloaded and cleaned:

```bash
python3 train_ml_models_real.py
```

This will:
1. Load all cleaned real datasets
2. Combine them into single training set
3. Train models for diabetes, CVD, hypertension
4. Save models to `models/real_data/`

### **Step 3: Compare Real vs Synthetic Models**

```bash
python3 compare_real_vs_synthetic.py
```

This shows:
- Performance comparison (ROC-AUC)
- Which data source produces better models
- Sample size differences
- Feature importance differences

---

## 📁 Directory Structure

```
Patient_Digital_Twin_Systems/
├── data/
│   ├── synthetic/              # 5M synthetic patients (2.8 GB)
│   └── real/                   # Real datasets
│       ├── raw/                # Downloaded (original)
│       ├── cleaned/            # Cleaned (ready for training)
│       ├── lineage.json        # Data provenance
│       └── daily_report.json   # Latest acquisition report
├── models/
│   ├── diabetes_model.pkl      # Synthetic data model
│   ├── cvd_model.pkl           # Synthetic data model
│   └── real_data/              # Real data models
│       ├── diabetes_model_real.pkl
│       ├── cvd_model_real.pkl
│       └── hypertension_model_real.pkl
└── logs/
    └── daily_acquisition.log   # Acquisition logs
```

---

## 🔄 Workflow

### **Daily Automated Acquisition**

```
2 AM Daily:
  ↓
Search repositories (Figshare, Zenodo, etc.)
  ↓
Download new datasets (up to 10GB)
  ↓
Validate data quality
  ↓
Clean and prepare
  ↓
Update lineage tracking
  ↓
Generate daily report
```

### **Training Workflow**

```
Real datasets downloaded
  ↓
Load and combine all cleaned datasets
  ↓
Standardize features (age, BMI, glucose, etc.)
  ↓
Train Gradient Boosting models
  ↓
Evaluate performance
  ↓
Save models + metadata
  ↓
Compare with synthetic models
```

---

## 📊 Expected Timeline

**Hour 1-2:** Data acquisition running
- Searching repositories
- Downloading datasets
- Finding 100-200 datasets

**Hour 2-3:** Validation and cleaning
- Checking data quality
- Removing duplicates
- Standardizing formats

**Hour 3+:** Ready for training
- 50-100 cleaned datasets available
- Combined: 10K-100K real patients
- Ready to train models

---

## 🎯 Commands Reference

### **Check Acquisition Status**
```bash
# View real-time logs
tail -f logs/daily_acquisition.log

# Check downloaded files
ls -lh data/real/raw/

# Check cleaned files
ls -lh data/real/cleaned/

# View statistics
python3 -c "
from data_engine.real_data_pipeline import RealDataPipeline
pipeline = RealDataPipeline()
stats = pipeline.lineage.get_statistics()
print(f'Total datasets: {stats[\"total_datasets\"]}')
print(f'Total size: {stats[\"total_size_gb\"]:.2f} GB')
print(f'Validated: {stats[\"validated_datasets\"]}')
print(f'Cleaned: {stats[\"cleaned_datasets\"]}')
"
```

### **Train Models**
```bash
# Train on real data
python3 train_ml_models_real.py

# Train on synthetic data (for comparison)
python3 train_ml_models.py
```

### **Compare Models**
```bash
# Compare performance
python3 compare_real_vs_synthetic.py

# View comparison report
cat reports/model_comparison.json | jq
```

---

## 🎓 What to Expect

### **Real Data Advantages:**
- ✅ Real patient outcomes
- ✅ Real-world edge cases
- ✅ Clinical validation possible
- ✅ Production-ready models

### **Real Data Challenges:**
- ⚠️ Smaller sample sizes (typically 10K-100K)
- ⚠️ Missing values
- ⚠️ Inconsistent formats
- ⚠️ Privacy considerations

### **Synthetic Data Advantages:**
- ✅ Massive scale (5M patients)
- ✅ Perfect data quality
- ✅ No privacy issues
- ✅ Controlled distributions

### **Synthetic Data Limitations:**
- ⚠️ Not real patients
- ⚠️ May miss rare patterns
- ⚠️ Cannot validate on real outcomes

---

## 🎯 Recommended Approach

**Hybrid Strategy:**

1. **Develop on synthetic** (fast iteration, large scale)
2. **Validate on real** (clinical accuracy)
3. **Deploy with both** (ensemble models)

```python
# Ensemble prediction
synthetic_pred = synthetic_model.predict_proba(X)
real_pred = real_model.predict_proba(X)

# Weighted average
final_pred = 0.6 * real_pred + 0.4 * synthetic_pred
```

---

## 📈 Next Steps

**Immediate (while acquisition runs):**
1. Monitor acquisition logs
2. Review downloaded datasets
3. Prepare training environment

**After acquisition completes:**
1. Train models on real data
2. Compare with synthetic models
3. Evaluate on test sets
4. Deploy best-performing models

**Long-term:**
1. Continuous daily acquisition
2. Automated model retraining
3. A/B testing real vs synthetic
4. Clinical validation studies

---

## 🛠️ Troubleshooting

**Acquisition not finding datasets:**
- Check internet connection
- Verify API endpoints
- Try different keywords

**Training fails:**
- Check if datasets downloaded
- Verify data format
- Review feature mappings

**Low model performance:**
- Check sample size
- Review data quality
- Adjust hyperparameters

---

## 📝 Summary

**You now have:**
- ✅ Automated real data acquisition (running now)
- ✅ ML training pipeline for real data
- ✅ Comparison tool (real vs synthetic)
- ✅ Complete data lineage tracking

**Next actions:**
1. Wait for acquisition to complete (~2 hours)
2. Run `python3 train_ml_models_real.py`
3. Run `python3 compare_real_vs_synthetic.py`
4. Deploy best models to production
