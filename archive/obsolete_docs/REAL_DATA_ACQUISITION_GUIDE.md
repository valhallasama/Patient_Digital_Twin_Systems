# Real Data Acquisition System

## Overview

Automated system for discovering, downloading, validating, and tracking **real health datasets** from public repositories.

**NO synthetic data** - only real patient data from public sources.

---

## 🎯 What It Does

### **1. Automated Daily Search**
Searches these public repositories every day:
- **Figshare** - Research datasets
- **Zenodo** - Scientific data
- **Data.gov** - Government health data
- **Kaggle** - ML datasets
- **PhysioNet** - Medical databases (requires credentials)
- **MIMIC-III** - ICU data (requires credentials)

### **2. Automated Download**
- Downloads all discovered health datasets
- Configurable size limit (default: 10GB/day)
- Respects rate limits and API quotas

### **3. Validation**
- Checks file integrity
- Validates data format
- Verifies column structure
- Identifies data quality issues

### **4. Cleaning**
- Removes duplicates
- Handles missing values
- Standardizes formats
- Prepares for ML training

### **5. Lineage Tracking**
- Tracks data source
- Records download timestamp
- Calculates file hash
- Maintains complete provenance

---

## 🚀 Quick Start

### **Option 1: Run Once (Manual)**

```bash
# Run data acquisition now
python3 run_daily_data_acquisition.py
```

This will:
1. Search all public repositories
2. Download up to 10GB of datasets
3. Validate all downloads
4. Clean valid datasets
5. Generate report

### **Option 2: Setup Automated Daily Runs**

```bash
# Setup cron job for daily 2 AM runs
bash setup_daily_acquisition.sh
```

The system will automatically run every day at 2 AM.

---

## 📊 Data Storage Structure

```
data/real/
├── raw/                          # Downloaded datasets (original)
│   ├── figshare_dataset1.csv
│   ├── zenodo_dataset2.xlsx
│   └── ...
├── cleaned/                      # Cleaned datasets (ready for training)
│   ├── figshare_dataset1_cleaned.csv
│   ├── zenodo_dataset2_cleaned.csv
│   └── ...
├── lineage.json                  # Complete data lineage tracking
└── daily_report.json            # Latest acquisition report
```

---

## 🔍 Search Keywords

The system searches for datasets related to:

**Diseases:**
- Diabetes
- Cardiovascular disease
- Hypertension
- Obesity
- Metabolic syndrome
- Stroke
- Chronic kidney disease
- Cancer
- COPD
- Asthma

**Data Types:**
- Patient records
- Clinical trials
- EHR (Electronic Health Records)
- Medical imaging
- Lab results
- Vital signs
- Wearable data

**Populations:**
- Population health
- Epidemiology
- Public health
- Cohort studies

**Specific Datasets:**
- NHANES
- UK Biobank
- MIMIC
- eICU
- ADNI
- TCGA

---

## 📈 Usage Examples

### **Example 1: Run Manual Acquisition**

```bash
python3 run_daily_data_acquisition.py
```

**Output:**
```
AUTOMATED DAILY DATA ACQUISITION
================================================================================
Searching all public repositories...
  Figshare: Found 45 datasets
  Zenodo: Found 32 datasets
  Data.gov: Found 28 datasets
  Kaggle: Found 19 datasets

Downloading datasets...
  [1/124] Diabetes Patient Records (Figshare) - 125 MB ✓
  [2/124] CVD Clinical Trial Data (Zenodo) - 89 MB ✓
  ...

Validating datasets...
  Valid: 98
  Invalid: 12
  Errors: 14

Cleaning datasets...
  Cleaned: 98 datasets

DAILY ACQUISITION SUMMARY
================================================================================
  Datasets found: 124
  Datasets downloaded: 98
  Datasets validated: 98
  Datasets cleaned: 98
  Repository total: 98
  Repository size: 8.45 GB
```

### **Example 2: Check Acquisition Status**

```python
from data_engine.real_data_pipeline import RealDataPipeline

pipeline = RealDataPipeline()

# Get statistics
stats = pipeline.lineage.get_statistics()

print(f"Total datasets: {stats['total_datasets']}")
print(f"Total size: {stats['total_size_gb']:.2f} GB")
print(f"Validated: {stats['validated_datasets']}")
print(f"Cleaned: {stats['cleaned_datasets']}")

# Get all datasets
datasets = pipeline.lineage.get_all_datasets(validated_only=True)

for ds in datasets[:5]:
    print(f"\n{ds['title']}")
    print(f"  Source: {ds['source']}")
    print(f"  Size: {ds['file_size'] / (1024**2):.1f} MB")
    print(f"  Downloaded: {ds['downloaded_at']}")
```

### **Example 3: Search Specific Topics**

```python
from data_engine.real_data_pipeline import RealDataPipeline

pipeline = RealDataPipeline()

# Search for specific disease
datasets = pipeline.search_all_sources(
    keywords=['diabetes', 'glucose', 'HbA1c'],
    limit_per_source=50
)

print(f"Found {len(datasets)} diabetes-related datasets")

# Download top 10
for dataset in datasets[:10]:
    pipeline.download_dataset(dataset)
```

### **Example 4: Get Data Lineage**

```python
from data_engine.real_data_pipeline import RealDataPipeline

pipeline = RealDataPipeline()

# Get info about specific dataset
dataset_info = pipeline.lineage.get_dataset_info('figshare_10.1234_dataset')

print(f"Title: {dataset_info['title']}")
print(f"Source: {dataset_info['source']}")
print(f"Downloaded: {dataset_info['downloaded_at']}")
print(f"File hash: {dataset_info['file_hash']}")
print(f"Validated: {dataset_info['validation_status']}")
print(f"Cleaned: {dataset_info['cleaned']}")
```

---

## 🎛️ Configuration

### **Adjust Download Limits**

Edit `run_daily_data_acquisition.py`:

```python
# Change max download size
result = pipeline.run_daily_acquisition(
    keywords=keywords,
    max_download_gb=20.0  # Download up to 20GB per day
)
```

### **Add More Keywords**

Edit `run_daily_data_acquisition.py`:

```python
keywords = [
    'diabetes',
    'cardiovascular disease',
    # Add your keywords here
    'mental health',
    'depression',
    'anxiety'
]
```

### **Change Schedule**

Edit cron job:

```bash
crontab -e

# Change from 2 AM to 10 PM:
0 22 * * * cd /path/to/project && python3 run_daily_data_acquisition.py
```

---

## 📊 Data Lineage Tracking

Every downloaded dataset is tracked with:

```json
{
  "dataset_id": "figshare_10.1234_diabetes_data",
  "source": "figshare",
  "title": "Diabetes Patient Records 2020-2023",
  "url": "https://figshare.com/articles/...",
  "doi": "10.1234/figshare.12345",
  "downloaded_at": "2024-03-11T14:30:00",
  "file_path": "data/real/raw/figshare_diabetes.csv",
  "file_size": 125829120,
  "file_hash": "sha256:abc123...",
  "validation_status": "valid",
  "validated_at": "2024-03-11T14:31:00",
  "cleaned": true,
  "cleaned_path": "data/real/cleaned/figshare_diabetes_cleaned.csv",
  "cleaned_at": "2024-03-11T14:32:00"
}
```

---

## 🔒 Data Sources & Credentials

### **Public (No Credentials Required):**
- ✅ Figshare
- ✅ Zenodo
- ✅ Data.gov
- ✅ Kaggle (API key optional)

### **Requires Registration:**
- ⚠️ PhysioNet (free registration)
- ⚠️ MIMIC-III (requires training + credentialing)
- ⚠️ UK Biobank (requires application)

**To add credentials:**

Edit `config/data_sources.yaml`:

```yaml
research_repositories:
  - name: "PhysioNet"
    api_url: "https://physionet.org/api"
    api_key: "your-api-key-here"
    requires_auth: true
```

---

## 📈 Expected Results

### **Daily Acquisition (10GB limit):**
- **Datasets found:** 100-200
- **Datasets downloaded:** 50-100
- **Datasets validated:** 45-95
- **Datasets cleaned:** 45-95

### **After 1 Week:**
- **Total datasets:** 300-700
- **Total size:** 50-70 GB
- **Validated datasets:** 250-650

### **After 1 Month:**
- **Total datasets:** 1,000-3,000
- **Total size:** 200-300 GB
- **Validated datasets:** 900-2,800

---

## 🎯 Next Steps After Data Acquisition

### **1. Train ML Models on Real Data**

```bash
# Train on real datasets instead of synthetic
python3 train_ml_models_real.py
```

### **2. Combine Multiple Datasets**

```python
from data_engine.real_data_pipeline import RealDataPipeline
import pandas as pd

pipeline = RealDataPipeline()

# Get all cleaned datasets
datasets = pipeline.lineage.get_all_datasets(validated_only=True)

# Load and combine
dfs = []
for ds in datasets:
    if ds.get('cleaned'):
        df = pd.read_csv(ds['cleaned_path'])
        dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset: {len(combined)} patients")
```

### **3. Monitor Data Quality**

```bash
# View daily reports
cat data/real/daily_report.json

# Check logs
tail -f logs/daily_acquisition.log
```

---

## 🛠️ Troubleshooting

### **No datasets found:**
- Check internet connection
- Verify API endpoints are accessible
- Try different keywords

### **Download failures:**
- Check disk space
- Verify file permissions
- Check rate limits

### **Validation failures:**
- Review validation logs
- Check file formats
- Verify data integrity

---

## 📝 Summary

**The system now:**
- ✅ Automatically searches public repositories daily
- ✅ Downloads real health datasets (not synthetic)
- ✅ Validates all downloaded data
- ✅ Cleans and prepares data for training
- ✅ Tracks complete data lineage
- ✅ Generates daily reports

**Run it now:**
```bash
python3 run_daily_data_acquisition.py
```

**Or setup automated daily runs:**
```bash
bash setup_daily_acquisition.sh
```
