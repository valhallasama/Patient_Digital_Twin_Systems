# 6-Month Research-Grade Digital Twin Development Plan

## 📋 Overview

**Goal:** Build a scientifically rigorous, literature-grounded, data-validated patient digital twin system

**Timeline:** 6 months (26 weeks)  
**Data:** MIMIC-III (40,000+ ICU patients)  
**Validation:** Multiple independent cohorts  
**Outcome:** Publishable research-grade model

---

## 📅 Detailed Timeline

### **PHASE 1: Data Access (Weeks 1-4)**

#### **Week 1: CITI Training**

**Tasks:**
- [ ] Register at https://www.citiprogram.org/
- [ ] Complete "Data or Specimens Only Research" course
- [ ] Modules required:
  - Belmont Report and Its Principles
  - History and Ethical Principles
  - Defining Research with Human Subjects
  - Research with Protected Populations
  - Informed Consent
  - Privacy and Confidentiality
  - Records-Based Research
- [ ] Download completion certificate

**Time:** 4-6 hours  
**Deliverable:** CITI completion certificate

**Resources:**
- CITI training: https://physionet.org/about/citi-course/
- PhysioNet guide: https://mimic.mit.edu/docs/gettingstarted/

---

#### **Week 2: PhysioNet Application**

**Tasks:**
- [ ] Create PhysioNet account at https://physionet.org/
- [ ] Complete credentialing application:
  - Upload CITI certificate
  - Provide institutional affiliation
  - Research use statement
  - Supervisor information (if student)
- [ ] Submit application
- [ ] Wait for approval (typically 1-2 weeks)

**Time:** 2-3 hours application, 1-2 weeks wait  
**Deliverable:** PhysioNet credentialing approval

**Note:** While waiting for approval, start literature review (Phase 2)

---

#### **Week 3-4: Download and Setup**

**Tasks:**
- [ ] Once approved, sign MIMIC-III Data Use Agreement
- [ ] Download MIMIC-III v1.4 (~50 GB compressed)
  ```bash
  wget -r -N -c -np --user USERNAME --ask-password \
    https://physionet.org/files/mimiciii/1.4/
  ```
- [ ] Install PostgreSQL database
  ```bash
  sudo apt-get install postgresql postgresql-contrib
  ```
- [ ] Load MIMIC-III into PostgreSQL
  ```bash
  git clone https://github.com/MIT-LCP/mimic-code.git
  cd mimic-code/buildmimic/postgres
  make mimic-gz datadir=/path/to/mimic/data
  ```
- [ ] Verify database integrity
- [ ] Run test queries to ensure proper setup

**Time:** 1-2 days download, 1 day setup  
**Deliverable:** Fully functional MIMIC-III database

**Database Schema:**
- 26 tables
- Key tables: PATIENTS, ADMISSIONS, LABEVENTS, CHARTEVENTS, DIAGNOSES_ICD
- ~40,000 patients, 58,000 admissions, 330 million lab events

---

### **PHASE 2: Literature Review (Weeks 2-14)**

**Parallel with Phase 1 - Start during waiting period**

#### **Week 2-4: Metabolic System (50 papers)**

**Parameters to Extract:**

**1. Beta Cell Function Decline**
- [ ] UKPDS 16 (Diabetes 1995) - Beta cell function in type 2 diabetes
- [ ] ADOPT trial (Diabetes Care 2006) - Durability of glycemic control
- [ ] Kahn SE et al. (Diabetes 2006) - Mechanisms of beta cell failure
- [ ] Butler AE et al. (Diabetes 2003) - Beta cell deficit in type 2 diabetes
- [ ] 10+ additional longitudinal studies

**Extract:**
- Decline rate: X% per year
- Factors affecting decline (age, BMI, glucose)
- Heterogeneity (SD, confidence intervals)

**2. Insulin Sensitivity**
- [ ] DeFronzo RA (Diabetes 2004) - Pathogenesis of type 2 diabetes
- [ ] Bergman RN (Diabetes 2005) - Minimal model
- [ ] HOMA-IR validation studies (5+ papers)
- [ ] Exercise effects on insulin sensitivity (10+ papers)
- [ ] Dietary effects (10+ papers)

**Extract:**
- Normal values by age/BMI
- Decline rates with obesity
- Improvement with interventions

**3. Glucose Metabolism**
- [ ] Meal response curves (10+ papers)
- [ ] GLUT4 translocation studies (5+ papers)
- [ ] Incretin effects (5+ papers)
- [ ] Hepatic glucose production (5+ papers)

**Deliverable:** 
- Evidence table: Metabolic_Parameters.xlsx
- 50+ papers reviewed with extracted parameters
- All citations in BibTeX format

---

#### **Week 5-7: Cardiovascular System (50 papers)**

**Parameters to Extract:**

**1. Atherosclerosis Progression**
- [ ] MESA study (Multi-Ethnic Study of Atherosclerosis)
  - Carotid IMT progression rates
  - Coronary calcium scoring
- [ ] Framingham Offspring Study
- [ ] Rotterdam Study
- [ ] 15+ imaging studies

**Extract:**
- IMT progression: mm/year by age, risk factors
- Plaque burden increase
- Regression with statins

**2. Blood Pressure**
- [ ] Franklin SS (Circulation 1997) - Age-related BP changes
- [ ] JNC 8 Guidelines (JAMA 2014)
- [ ] SPRINT trial (NEJM 2015) - Intensive BP control
- [ ] Exercise effects on BP (10+ meta-analyses)
- [ ] Dietary effects (DASH, sodium) (10+ papers)

**Extract:**
- Age-related SBP increase: mmHg/year
- DBP changes with age
- Intervention effects (exercise, diet, meds)

**3. Vessel Elasticity**
- [ ] Pulse wave velocity studies (10+ papers)
- [ ] Arterial stiffness indices (10+ papers)
- [ ] Age-related changes (5+ papers)

**Deliverable:**
- Evidence table: Cardiovascular_Parameters.xlsx
- 50+ papers with extracted parameters

---

#### **Week 8-10: Renal System (50 papers)**

**Parameters to Extract:**

**1. eGFR Decline**
- [ ] KDIGO 2012 Guidelines - CKD classification
- [ ] Coresh et al. (JASN 2014) - Decline in general population
- [ ] Perkins et al. (Diabetes Care 2003) - Diabetic nephropathy
- [ ] RENAAL trial - ARB effects on CKD
- [ ] IDNT trial
- [ ] 20+ longitudinal CKD studies

**Extract:**
- Normal aging: -1 mL/min/1.73m²/year after age 40
- CKD Stage 3: -3 to -5 mL/min/year
- Diabetic nephropathy: -5 to -10 mL/min/year
- Intervention effects (BP control, RAAS blockade)

**2. Proteinuria**
- [ ] Progression rates (10+ papers)
- [ ] Relationship to eGFR decline (10+ papers)

**3. Tubular Function**
- [ ] Creatinine clearance (5+ papers)
- [ ] Tubular markers (5+ papers)

**Deliverable:**
- Evidence table: Renal_Parameters.xlsx
- 50+ papers with decline rates by subgroup

---

#### **Week 11-13: Other Organ Systems (50 papers)**

**Hepatic System (15 papers):**
- [ ] NAFLD progression (5 papers)
- [ ] Lipid metabolism (5 papers)
- [ ] Liver enzyme changes (5 papers)

**Immune/Inflammatory (15 papers):**
- [ ] Cytokine dynamics (5 papers)
- [ ] Chronic inflammation (5 papers)
- [ ] Age-related changes (5 papers)

**Endocrine System (10 papers):**
- [ ] HPA axis function (3 papers)
- [ ] Cortisol dynamics (4 papers)
- [ ] Thyroid function (3 papers)

**Neural System (10 papers):**
- [ ] Stress response (3 papers)
- [ ] Sleep effects on metabolism (4 papers)
- [ ] Cognitive changes (3 papers)

**Deliverable:**
- Evidence tables for each system
- 50+ papers total

---

#### **Week 14: Consolidate Evidence**

**Tasks:**
- [ ] Merge all evidence tables
- [ ] Create master parameter database
- [ ] Document all citations in reference manager
- [ ] Write literature review summary document
- [ ] Identify gaps and uncertainties
- [ ] Create parameter ranges (mean ± SD)

**Deliverable:**
- Master_Evidence_Table.xlsx (200+ papers, 500+ parameters)
- Literature_Review_Summary.pdf (30-50 pages)
- parameters.json (machine-readable parameter database)

**Format:**
```json
{
  "egfr_decline": {
    "normal_aging": {
      "value": -1.0,
      "unit": "mL/min/1.73m²/year",
      "sd": 0.5,
      "source": "Coresh et al. JASN 2014",
      "pmid": "24578131",
      "sample_size": 15000,
      "age_range": "40-80"
    },
    "ckd_stage_3": {
      "value": -4.0,
      "unit": "mL/min/1.73m²/year",
      "sd": 2.0,
      "source": "KDIGO 2012",
      "sample_size": "meta-analysis"
    }
  }
}
```

---

### **PHASE 3: Data Preprocessing (Weeks 15-18)**

#### **Week 15: Cohort Extraction**

**Task 1: Diabetes Cohort**
```sql
-- Extract patients with diabetes diagnosis
SELECT DISTINCT p.subject_id, p.gender, p.dob,
       a.admittime, a.dischtime,
       d.icd9_code, d.short_title
FROM patients p
INNER JOIN admissions a ON p.subject_id = a.subject_id
INNER JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
WHERE d.icd9_code LIKE '250%'  -- Diabetes codes
   OR d.icd9_code IN ('E10%', 'E11%')  -- ICD-10 codes
```

**Extract:**
- [ ] Identify ~8,000 diabetes patients
- [ ] Get all lab values (glucose, HbA1c, creatinine)
- [ ] Get all vitals (BP, HR, weight)
- [ ] Get medications (insulin, metformin, etc.)
- [ ] Get outcomes (mortality, complications)

**Task 2: CKD Cohort**
```sql
-- Extract patients with CKD (eGFR < 60)
SELECT subject_id, charttime, valuenum as creatinine
FROM labevents
WHERE itemid IN (50912)  -- Creatinine
  AND valuenum > 1.5  -- Elevated creatinine
```

**Extract:**
- [ ] ~5,000 CKD patients
- [ ] Serial creatinine measurements
- [ ] Calculate eGFR using CKD-EPI equation
- [ ] Track progression over time

**Task 3: CVD Cohort**
```sql
-- Extract patients with cardiovascular disease
WHERE icd9_code LIKE '410%'  -- MI
   OR icd9_code LIKE '411%'  -- Angina
   OR icd9_code LIKE '428%'  -- Heart failure
```

**Extract:**
- [ ] ~10,000 CVD patients
- [ ] Cardiac biomarkers (troponin, BNP)
- [ ] Echo data if available
- [ ] Outcomes

**Deliverable:**
- diabetes_cohort.csv (~8,000 patients)
- ckd_cohort.csv (~5,000 patients)
- cvd_cohort.csv (~10,000 patients)

---

#### **Week 16: Feature Engineering**

**Tasks:**
- [ ] Calculate eGFR from creatinine (CKD-EPI equation)
  ```python
  def calculate_egfr(creatinine, age, sex, race):
      # CKD-EPI equation
      kappa = 0.7 if sex == 'F' else 0.9
      alpha = -0.329 if sex == 'F' else -0.411
      # ... full equation
      return egfr
  ```

- [ ] Derive time-dependent features:
  - Days since admission
  - Days since diagnosis
  - Cumulative medication exposure
  - Rate of change (slopes)

- [ ] Create interaction features:
  - Age × diabetes
  - BMI × hypertension
  - Glucose × creatinine

- [ ] Temporal aggregations:
  - Mean glucose over 30 days
  - Max BP in last week
  - Trend in eGFR

**Deliverable:**
- feature_engineering.py
- Expanded dataset with 50+ features per patient

---

#### **Week 17: Missing Data Handling**

**MIMIC-III has 30-70% missingness - need proper handling**

**Strategy:**
1. **Characterize missingness**
   - [ ] Calculate missingness rate per variable
   - [ ] Identify patterns (MCAR, MAR, MNAR)
   - [ ] Visualize missingness patterns

2. **Multiple imputation**
   ```python
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer
   
   # MICE (Multiple Imputation by Chained Equations)
   imputer = IterativeImputer(max_iter=10, random_state=42)
   imputed_data = imputer.fit_transform(data)
   ```

3. **Sensitivity analysis**
   - [ ] Compare results with different imputation methods
   - [ ] Assess impact on parameter estimates

**Deliverable:**
- missing_data_report.pdf
- Imputed datasets (5 imputations for uncertainty)

---

#### **Week 18: Quality Control**

**Tasks:**
- [ ] Outlier detection
  - Physiologically implausible values
  - Statistical outliers (>3 SD)
  - Expert review of extreme values

- [ ] Data validation
  - Cross-check against known distributions
  - Verify temporal consistency
  - Check for data entry errors

- [ ] Create clean dataset
  - Document exclusions
  - Final cohort characteristics table

**Deliverable:**
- clean_mimic_data.csv (30,000+ patients after QC)
- data_quality_report.pdf
- CONSORT-style flow diagram

---

### **PHASE 4: Model Development (Weeks 19-26)**

#### **Week 19-20: Parameter Estimation**

**Task 1: Population-Level Parameters**

Use mixed-effects models to estimate organ decline rates:

```python
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# eGFR decline rate
model = MixedLM.from_formula(
    'egfr ~ age + diabetes + hypertension + baseline_egfr + time',
    data=longitudinal_data,
    groups=longitudinal_data['patient_id'],
    re_formula='~time'  # Random slope for time
)

result = model.fit()
print(result.summary())

# Extract parameters
egfr_decline_rate = result.params['time']
egfr_decline_se = result.bse['time']
```

**Estimate for each organ:**
- [ ] Metabolic: HbA1c progression, beta cell decline
- [ ] Cardiovascular: BP increase, atherosclerosis
- [ ] Renal: eGFR decline by subgroup
- [ ] Hepatic: ALT changes, lipid metabolism

**Task 2: Interaction Parameters**

```python
# How does glucose affect kidney function?
model = MixedLM.from_formula(
    'egfr ~ glucose + glucose:time + ...',
    ...
)
```

**Deliverable:**
- empirical_parameters.json (all parameters with CI)
- parameter_estimation_report.pdf

---

#### **Week 21-22: Physics-Informed Neural Network**

**Implement PINN to enforce physiological constraints:**

```python
import torch
import torch.nn as nn

class PhysicsInformedPatientModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Neural network for learning patterns
        self.lstm = nn.LSTM(input_size=50, hidden_size=256, num_layers=3)
        self.fc = nn.Linear(256, 6)  # Predict 6 lab values
        
    def forward(self, x):
        # Standard prediction
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        
        return predictions
    
    def physics_loss(self, predictions, inputs):
        """
        Enforce physiological constraints
        """
        glucose, hba1c, egfr, creatinine, sbp, dbp = predictions.unbind(-1)
        
        # Constraint 1: HbA1c correlates with glucose
        # HbA1c ≈ (glucose + 2.59) / 1.59 (ADAG formula)
        hba1c_expected = (glucose + 2.59) / 1.59
        hba1c_loss = torch.mean((hba1c - hba1c_expected)**2)
        
        # Constraint 2: Creatinine inversely related to eGFR
        # eGFR ≈ 175 × creatinine^(-1.154) × age^(-0.203)
        age = inputs[:, :, 0]  # Assuming age is first feature
        egfr_expected = 175 * (creatinine ** -1.154) * (age ** -0.203)
        egfr_loss = torch.mean((egfr - egfr_expected)**2)
        
        # Constraint 3: Glucose must be positive
        glucose_positive_loss = torch.mean(torch.relu(-glucose))
        
        # Constraint 4: eGFR can't increase rapidly (max +5 mL/min/year)
        egfr_change = egfr[:, 1:] - egfr[:, :-1]
        egfr_increase_loss = torch.mean(torch.relu(egfr_change - 5.0/365))
        
        # Total physics loss
        physics_loss = (hba1c_loss + egfr_loss + 
                       glucose_positive_loss + egfr_increase_loss)
        
        return physics_loss

# Training loop
def train_pinn(model, data_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            
            # Predictions
            predictions = model(batch_x)
            
            # Data loss (MSE)
            data_loss = nn.MSELoss()(predictions, batch_y)
            
            # Physics loss
            physics_loss = model.physics_loss(predictions, batch_x)
            
            # Combined loss
            total_loss = data_loss + 0.1 * physics_loss  # Weight physics term
            
            total_loss.backward()
            optimizer.step()
```

**Deliverable:**
- pinn_model.py
- Trained PINN model
- Validation showing physics constraints are satisfied

---

#### **Week 23-24: LSTM with Hyperparameter Optimization**

**Proper architecture search:**

```python
import optuna

def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Build model
    model = PatientLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Train
    val_loss = train_and_validate(model, learning_rate, batch_size)
    
    return val_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)
```

**Train final model:**
- 30,000 patients training
- 5,000 validation
- 5,000 test
- 5-fold cross-validation

**Deliverable:**
- Optimized LSTM model
- Hyperparameter search results
- Cross-validation performance

---

#### **Week 25-26: Hybrid Integration**

**Combine parametric + PINN + LSTM:**

```python
class HybridDigitalTwin:
    def __init__(self):
        self.parametric_model = ParametricOrganModel()  # Literature-based
        self.pinn = PhysicsInformedNN()  # Physics constraints
        self.lstm = OptimizedLSTM()  # Pattern learning
        
    def predict(self, patient_history):
        # Parametric prediction
        param_pred = self.parametric_model.predict(patient_history)
        
        # PINN prediction (enforces physics)
        pinn_pred = self.pinn.predict(patient_history)
        
        # LSTM prediction (learns patterns)
        lstm_pred = self.lstm.predict(patient_history)
        
        # Ensemble (learned weights)
        final_pred = (0.3 * param_pred + 
                     0.3 * pinn_pred + 
                     0.4 * lstm_pred)
        
        return final_pred
```

**Deliverable:**
- Fully integrated hybrid model
- Ensemble weights optimized on validation set

---

### **PHASE 5: Validation (Weeks 27-30)**

#### **Week 27: Internal Validation**

**Tasks:**
- [ ] Evaluate on held-out test set (5,000 patients)
- [ ] Calculate metrics:
  - MAE, RMSE for each biomarker
  - R² for predictions
  - Calibration plots
  - Prediction intervals (95% CI)

**Metrics to report:**
```python
from sklearn.metrics import mean_absolute_error, r2_score

# For each biomarker
for biomarker in ['glucose', 'hba1c', 'egfr', 'sbp']:
    mae = mean_absolute_error(y_true[biomarker], y_pred[biomarker])
    r2 = r2_score(y_true[biomarker], y_pred[biomarker])
    
    print(f"{biomarker}: MAE={mae:.3f}, R²={r2:.3f}")
```

**Deliverable:**
- internal_validation_report.pdf
- Performance metrics table

---

#### **Week 28: Temporal Validation**

**Test on different time periods:**

```python
# Train on 2008-2012 data
train_data = mimic_data[mimic_data['year'] <= 2012]

# Test on 2013-2016 data
test_data = mimic_data[mimic_data['year'] > 2012]

# Check for distribution shift
```

**Deliverable:**
- Temporal validation results
- Distribution shift analysis

---

#### **Week 29: Comparison to Clinical Risk Scores**

**Compare to established scores:**

**1. Framingham Risk Score (CVD)**
```python
def framingham_risk(age, sex, sbp, cholesterol, hdl, smoker, diabetes):
    # Framingham equation
    ...
    return 10_year_cvd_risk
```

**2. UKPDS Risk Engine (Diabetes complications)**

**3. KDIGO Risk Prediction (CKD progression)**

**Analysis:**
- [ ] Calculate C-statistic for each
- [ ] Compare discrimination
- [ ] Compare calibration
- [ ] Decision curve analysis

**Deliverable:**
- Comparison table showing our model vs. clinical scores
- ROC curves, calibration plots

---

#### **Week 30: Subgroup Analysis & Final Report**

**Subgroup performance:**
- By age (<50, 50-65, >65)
- By sex (male, female)
- By race/ethnicity
- By disease severity
- By comorbidities

**Final validation report:**
- [ ] Executive summary
- [ ] Methods (data, model, validation)
- [ ] Results (all metrics)
- [ ] Subgroup analysis
- [ ] Limitations
- [ ] Clinical implications
- [ ] Future work

**Deliverable:**
- Final_Validation_Report.pdf (50-100 pages)
- Supplementary materials
- Code repository
- Trained models

---

## 📊 Expected Outcomes

### **Performance Targets**

Based on literature, expect:

| Biomarker | Target MAE | Target R² |
|-----------|-----------|-----------|
| Glucose | < 0.5 mmol/L | > 0.85 |
| HbA1c | < 0.3% | > 0.90 |
| eGFR | < 3 mL/min | > 0.85 |
| Creatinine | < 10 μmol/L | > 0.85 |
| SBP | < 5 mmHg | > 0.80 |
| DBP | < 3 mmHg | > 0.80 |

### **Comparison to Clinical Scores**

Expect to match or exceed:
- Framingham C-statistic: 0.75-0.80
- UKPDS C-statistic: 0.70-0.75
- KDIGO C-statistic: 0.75-0.80

---

## 📁 Deliverables Checklist

### **Data & Code**
- [ ] MIMIC-III database (local PostgreSQL)
- [ ] Cleaned datasets (CSV files)
- [ ] Feature engineering pipeline
- [ ] All model code (Python)
- [ ] Training scripts
- [ ] Evaluation scripts
- [ ] GitHub repository

### **Documentation**
- [ ] Literature review summary (30-50 pages)
- [ ] Master evidence table (200+ papers)
- [ ] Parameter database (JSON)
- [ ] Data quality report
- [ ] Model architecture documentation
- [ ] Final validation report (50-100 pages)

### **Models**
- [ ] Parametric organ model (literature-based)
- [ ] Physics-informed neural network
- [ ] Optimized LSTM
- [ ] Hybrid ensemble model
- [ ] All saved with weights

### **Publications**
- [ ] Draft manuscript for medical journal
- [ ] Supplementary materials
- [ ] Code and data availability statement

---

## 🎯 Success Criteria

**Minimum viable:**
- ✅ All 200+ papers reviewed with parameters extracted
- ✅ MIMIC-III data properly preprocessed
- ✅ Model trained on 30k+ patients
- ✅ Validation on independent 5k test set
- ✅ Performance comparable to clinical risk scores
- ✅ Full documentation and code

**Stretch goals:**
- 🎯 Performance exceeds clinical risk scores
- 🎯 External validation on different dataset
- 🎯 Accepted for publication in medical journal
- 🎯 Open-source release with documentation

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| PhysioNet approval delayed | High | Apply early, start literature review in parallel |
| Missing data too severe | Medium | Multiple imputation, sensitivity analysis |
| Model doesn't converge | Medium | Extensive hyperparameter tuning, simpler architecture |
| Performance below targets | High | Ensemble methods, more data, better features |
| Literature review too slow | Medium | Prioritize key papers, use systematic search |

---

## 📞 Support & Resources

**Technical:**
- MIMIC-III documentation: https://mimic.mit.edu/docs/
- MIMIC Code Repository: https://github.com/MIT-LCP/mimic-code
- PhysioNet forums: https://physionet.org/forums/

**Literature:**
- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- Google Scholar: https://scholar.google.com/
- Reference manager: Zotero, Mendeley, EndNote

**Computing:**
- GPU access (Google Colab, AWS, local)
- Storage for 50+ GB data
- PostgreSQL database

---

## 🚀 Getting Started

**Week 1 Action Items:**
1. Register for CITI training TODAY
2. Complete course (4-6 hours)
3. Create PhysioNet account
4. Start literature search for metabolic parameters
5. Set up reference manager (Zotero)

**Ready to begin?** Let me know when you've completed CITI training and I'll help with the PhysioNet application.
