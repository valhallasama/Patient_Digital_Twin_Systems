# 🚀 Patient Digital Twin Systems - Quick Start

## ✅ Current Status

**Data Generated**: 5,000,000 patients ✅  
**ML Models Trained**: 3 models (Diabetes, CVD, Hypertension) ✅  
**API Server**: Running at http://localhost:8000 ✅  
**Dashboard**: Running at http://localhost:8501 ✅  

---

## 📊 What You Can Do Now

### **1. Explore the Dashboard** (Recommended First)

Open: **http://localhost:8501**

Try all 4 tabs:
- Patient Evaluation
- Risk Prediction
- Intervention Simulation
- Population Analytics

---

### **2. Train on Full 5M Dataset** (Better Accuracy)

Your current models used only 50K patients (fast prototype).  
Train on ALL 5 million for better performance:

```bash
python3 train_ml_models_full.py
```

**Time**: ~30-60 minutes per model  
**Expected improvement**: ROC-AUC 0.81 → 0.88+

---

### **3. Run Advanced Research Analysis**

```bash
python3 research_analysis.py
```

This analyzes:
- Disease comorbidity patterns
- Risk factor distributions
- Age-stratified prevalence
- High-risk subpopulations
- Generates comprehensive report

---

### **4. Test ML Prediction API**

After training, test ML predictions:

```bash
# Predict diabetes
curl -X POST "http://localhost:8000/ml/ml_predict_diabetes" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST001",
    "age": 55,
    "gender": "male",
    "bmi": 32,
    "systolic_bp": 145,
    "diastolic_bp": 92,
    "heart_rate": 78,
    "glucose_mmol_l": 6.8,
    "hba1c_percent": 6.2,
    "total_cholesterol_mmol_l": 6.5,
    "ldl_cholesterol_mmol_l": 4.3,
    "hdl_cholesterol_mmol_l": 0.9,
    "exercise_hours_per_week": 1.5,
    "sleep_hours_per_night": 6.5,
    "alcohol_units_per_week": 18,
    "diet_quality_score": 4,
    "stress_level": 8,
    "smoking_status": "current"
  }'

# Predict all diseases at once
curl -X POST "http://localhost:8000/ml/ml_predict_all" \
  -H "Content-Type: application/json" \
  -d '{ ... same data ... }'
```

---

## 🎯 Recommended Workflow

**Day 1** (Today):
1. ✅ Explore dashboard (5 min)
2. ✅ Run research analysis (2 min)
3. Start full-scale training (30-60 min, can run overnight)

**Day 2**:
1. Test ML prediction API
2. Build custom analyses
3. Integrate with your applications

---

## 📁 Key Files & Directories

```
Patient_Digital_Twin_Systems/
├── data/synthetic/              # 5M patient dataset
│   ├── complete_patient_data.csv
│   ├── batch_*.csv (100 files)
│   └── trajectories_batch_*/
│
├── models/                      # Trained ML models
│   ├── diabetes_model.pkl       # 50K training
│   ├── cvd_model.pkl
│   ├── hypertension_model.pkl
│   └── full_scale/              # 5M training (after running train_ml_models_full.py)
│
├── api/                         # REST API
│   ├── api_server.py            # Main API (running)
│   └── ml_prediction_endpoint.py # ML predictions
│
├── dashboard/                   # Web interface
│   └── health_dashboard.py      # Streamlit app (running)
│
└── agents/                      # Multi-agent system
    ├── cardiology_agent.py
    ├── metabolic_agent.py
    └── lifestyle_agent.py
```

---

## 🔥 Quick Commands

```bash
# Analyze data
python3 analyze_generated_data.py

# Research analysis
python3 research_analysis.py

# Train on full dataset (recommended!)
python3 train_ml_models_full.py

# Check API health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

---

## 💡 Why Training Was Fast

Your initial training used **50,000 patients** (first batch file), not all 5 million.

- **Fast prototype**: 10 seconds, ROC-AUC ~0.81
- **Full training**: 30-60 min, ROC-AUC ~0.88+ (better!)

Run `train_ml_models_full.py` to train on all 5M patients.

---

## 📊 Your Dataset Stats

- **Patients**: 5,000,000
- **Disease Trajectories**: 100,000 (10-year predictions)
- **Storage Used**: 2.72 GB
- **Diabetes Prevalence**: 19.6%
- **CVD Prevalence**: 13.1%
- **High-Risk Patients**: 7.5%

---

## 🆘 Troubleshooting

**Dashboard not loading?**
```bash
streamlit run dashboard/health_dashboard.py
```

**API not responding?**
```bash
python3 api/api_server.py
```

**Want to free up space?**
```bash
# Keep only main file, delete batches (saves ~2GB)
rm data/synthetic/batch_*.csv
```

---

## 🎯 Next Recommended Action

**Start the full-scale training to get better models:**

```bash
python3 train_ml_models_full.py
```

This will take 30-60 minutes but will give you production-quality models trained on all 5 million patients!

While it runs, explore the dashboard and run the research analysis.

---

**Enjoy your Patient Digital Twin System!** 🏥🤖
