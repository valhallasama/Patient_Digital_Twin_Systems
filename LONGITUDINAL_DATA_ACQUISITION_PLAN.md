# Longitudinal Cohort Data Acquisition Plan

## Objective
Acquire real longitudinal organ trajectory data to convert current multi-organ risk calculator into true digital twin.

---

## Target Datasets

### 1. Framingham Heart Study (FHS) - HIGHEST PRIORITY

**Why Critical:**
- Gold standard for cardiovascular + metabolic trajectories
- 75+ years of follow-up data
- Same individuals measured every 2-4 years
- Complete organ panels: metabolic, cardiovascular, liver, kidney
- Lifestyle factors tracked: alcohol, smoking, diet, exercise, medications

**What We Get:**
- **Liver trajectories:** ALT, AST, GGT over decades (MISSING in NHANES)
- **Lifestyle trajectories:** Real alcohol/exercise/diet changes over time (MISSING in NHANES)
- **Metabolic trajectories:** Enhanced beyond NHANES (more time points)
- **Cardiovascular trajectories:** Enhanced beyond NHANES
- **Disease onset timing:** Exact dates of diabetes, CVD, fatty liver diagnosis

**Sample Size:**
- Original cohort: ~5,000 participants
- Offspring cohort: ~5,000 participants
- Third generation: ~4,000 participants
- Total: ~14,000 participants with multi-generational data

**Temporal Coverage:**
- Original cohort: 1948-present (75+ years)
- Offspring: 1971-present (50+ years)
- Typical follow-up: 10-20 exams per person

**Access Process:**
1. Register for dbGaP account (NIH)
2. Submit Data Access Request (DAR)
3. Institutional certification required
4. IRB approval required
5. Timeline: 2-3 months from submission to approval

**Cost:** Free for academic research

**Application Requirements:**
- Research proposal (2-3 pages)
- IRB approval letter
- Data security plan
- Institutional signing official

**Link:** https://dbgap.ncbi.nlm.nih.gov/aa/wga.cgi?page=login

---

### 2. UK Biobank - HIGHEST PRIORITY

**Why Critical:**
- Largest cohort with imaging + biomarkers
- 500,000 participants
- Repeated assessments (baseline + 2-4 follow-ups)
- Imaging substudy: MRI liver fat, cardiac function
- Linked hospital records: exact disease onset dates
- Medications tracked over time

**What We Get:**
- **Liver imaging:** Direct liver fat measurement (not just enzymes)
- **All organ systems:** Complete coverage
- **Medication effects:** Real-world drug impact on organ trajectories
- **Disease outcomes:** Hospital admissions, diagnoses, procedures
- **Genetic data:** Can add genetic risk factors

**Sample Size:**
- Total participants: 502,000
- Imaging substudy: 100,000
- Repeat assessments: ~50,000-100,000 (ongoing)

**Temporal Coverage:**
- Baseline: 2006-2010
- First repeat: 2012-2013
- Imaging: 2014-present
- Follow-up: 10-15 years so far

**Access Process:**
1. Register on UK Biobank Access Management System (AMS)
2. Submit application with research proposal
3. Material Transfer Agreement
4. Timeline: 1-2 months from submission to approval

**Cost:** 
- Application fee: £0 for academic research
- Data access: Free
- Compute resources: May need to pay for cloud compute

**Application Requirements:**
- Research proposal (3-5 pages)
- Ethical approval (can be obtained after data access)
- Institution registration
- Principal Investigator details

**Link:** https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access

---

### 3. 45 and Up Study - HIGH PRIORITY (Australia)

**Why Critical for You:**
- You are in Australia → easier ethics approval
- Linked to Australian Medicare, hospital, pathology records
- Real-world Australian population
- Longitudinal outcomes over 15+ years

**What We Get:**
- **Pathology data:** All lab tests from Medicare (glucose, ALT, creatinine, etc.)
- **Medication data:** PBS (Pharmaceutical Benefits Scheme) records
- **Hospital data:** Admissions, diagnoses, procedures
- **Lifestyle questionnaires:** Repeated surveys
- **Disease outcomes:** Real-world progression in Australian healthcare system

**Sample Size:**
- Total participants: 267,000
- Age: 45+ years
- Follow-up: 2006-present (17+ years)

**Temporal Coverage:**
- Baseline questionnaire: 2006-2009
- Follow-up questionnaires: Every 3-5 years
- Linked data: Continuous from 2006-present

**Access Process:**
1. Submit application to Sax Institute
2. Australian institution required
3. Ethics approval required (HREC)
4. Timeline: 1-2 months

**Cost:**
- Application fee: ~AUD $1,000-3,000 (varies by project)
- Data access: Additional fees for linked data
- Cheaper for Australian researchers

**Application Requirements:**
- Research proposal
- Ethics approval (HREC)
- Australian institution affiliation
- Data security plan

**Link:** https://www.saxinstitute.org.au/our-work/45-up-study/for-researchers/

---

### 4. MESA (Multi-Ethnic Study of Atherosclerosis) - MEDIUM PRIORITY

**Why Valuable:**
- Multi-ethnic cohort (generalizable)
- Detailed cardiovascular + metabolic + kidney markers
- Imaging + biomarkers
- Clean data quality

**What We Get:**
- **Cardiovascular trajectories:** Enhanced beyond NHANES
- **Metabolic trajectories:** Enhanced beyond NHANES
- **Kidney trajectories:** Enhanced beyond NHANES
- **Ethnic diversity:** Can model population differences

**Sample Size:**
- Total participants: ~6,800
- Ethnicities: White, Black, Hispanic, Chinese
- Follow-up exams: 6 exams over 18 years

**Temporal Coverage:**
- Baseline: 2000-2002
- Exams: Every 2-3 years
- Follow-up: 18+ years

**Access Process:**
- dbGaP (same as Framingham)
- Timeline: 2-3 months

**Cost:** Free for academic research

**Link:** https://www.mesa-nhlbi.org/

---

### 5. BLSA (Baltimore Longitudinal Study of Aging) - MEDIUM PRIORITY

**Why Valuable:**
- Longest running aging study
- 30-40 year follow-up per person
- Cognitive decline trajectories
- Organ aging patterns

**What We Get:**
- **Neural trajectories:** Cognitive decline over decades (MISSING in NHANES)
- **Aging patterns:** How all organs change with age
- **Metabolic aging:** Long-term glucose, insulin trajectories
- **Cardiovascular aging:** Long-term BP, cholesterol trajectories

**Sample Size:**
- Total participants: ~3,000
- Some followed for 40+ years

**Temporal Coverage:**
- Started: 1958
- Ongoing
- Visits: Every 1-4 years

**Access Process:**
- Apply to NIA (National Institute on Aging)
- Timeline: 2-3 months

**Cost:** Free for academic research

**Link:** https://www.nia.nih.gov/research/labs/blsa

---

## Data Acquisition Timeline

### Month 1-2: Application Preparation
- Week 1: Prepare research proposals (Framingham, UK Biobank, 45 and Up)
- Week 2: Obtain IRB/HREC approval
- Week 3: Prepare data security plans
- Week 4: Submit applications

### Month 2-4: Approval Process
- Framingham (dbGaP): 2-3 months review
- UK Biobank: 1-2 months review
- 45 and Up: 1-2 months review

### Month 4-5: Data Access
- Download datasets
- Set up secure computing environment
- Initial data exploration

### Month 5-6: Data Processing
- Harmonize across cohorts
- Extract temporal trajectories
- Quality control

### Month 6-9: Model Training
- Train temporal transformers on real trajectories
- Validate on held-out patients
- Compare to current NHANES-based models

---

## What Each Dataset Provides

| Organ System | NHANES (Current) | Framingham | UK Biobank | 45 and Up | MESA | BLSA |
|--------------|------------------|------------|------------|-----------|------|------|
| **Metabolic** | ✅ 33,994 transitions | ✅ 10,000+ | ✅ 50,000+ | ✅ 20,000+ | ✅ 6,000+ | ✅ 3,000+ |
| **Cardiovascular** | ✅ 33,994 transitions | ✅ 10,000+ | ✅ 50,000+ | ✅ 20,000+ | ✅ 6,000+ | ✅ 3,000+ |
| **Kidney** | ✅ 33,994 transitions | ✅ 10,000+ | ✅ 50,000+ | ✅ 20,000+ | ✅ 6,000+ | ✅ 3,000+ |
| **Liver** | ❌ Constant | ✅ 10,000+ | ✅ 50,000+ | ✅ 20,000+ | ⚠️ Limited | ⚠️ Limited |
| **Immune** | ❌ Constant | ⚠️ Limited | ✅ 50,000+ | ✅ 20,000+ | ⚠️ Limited | ⚠️ Limited |
| **Neural** | ❌ Constant | ⚠️ Limited | ✅ 20,000+ | ⚠️ Limited | ⚠️ Limited | ✅ 3,000+ |
| **Lifestyle** | ❌ Constant | ✅ 10,000+ | ✅ 50,000+ | ✅ 20,000+ | ✅ 6,000+ | ✅ 3,000+ |

---

## Recommended Strategy

### Phase 1: Framingham + UK Biobank (Priority)
**Focus:** Liver + Lifestyle trajectories

**Rationale:**
- These are the biggest gaps in current system
- Framingham has best liver data
- UK Biobank has largest sample size
- Combined: ~60,000+ liver trajectories

**Timeline:** 3-4 months to data access

### Phase 2: 45 and Up (Australian Context)
**Focus:** Real-world Australian population validation

**Rationale:**
- Validate models on Australian population
- Easier ethics approval for you
- Linked Medicare data provides real-world validation

**Timeline:** 2-3 months to data access (parallel with Phase 1)

### Phase 3: MESA + BLSA (Enhancement)
**Focus:** Ethnic diversity + Neural trajectories

**Rationale:**
- MESA adds ethnic diversity
- BLSA adds neural/cognitive trajectories
- Can be added after core system is working

**Timeline:** 3-4 months to data access (after Phase 1)

---

## Expected Outcomes

### After Framingham + UK Biobank

**Learned Organs (6/7):**
- ✅ Metabolic (NHANES + Framingham + UK Biobank)
- ✅ Cardiovascular (NHANES + Framingham + UK Biobank)
- ✅ Kidney (NHANES + Framingham + UK Biobank)
- ✅ **Liver (Framingham + UK Biobank)** ← NEW
- ✅ **Immune (UK Biobank)** ← NEW
- ✅ **Lifestyle (Framingham + UK Biobank)** ← NEW

**Placeholder Organs (1/7):**
- ⚠️ Neural (awaiting BLSA)

**Digital Twin Completeness: 85%**

### After All Datasets

**Learned Organs (7/7):**
- All organ systems learned from real trajectories
- Multi-cohort validation
- Ethnic diversity
- Australian population validation

**Digital Twin Completeness: 100%**

---

## Budget Estimate

| Item | Cost (AUD) |
|------|-----------|
| Framingham (dbGaP) | $0 |
| UK Biobank | $0 |
| 45 and Up application | $1,000-3,000 |
| MESA (dbGaP) | $0 |
| BLSA | $0 |
| Compute resources | $500-2,000 |
| **Total** | **$1,500-5,000** |

Very affordable for a high-impact publication.

---

## Next Steps (This Week)

1. **Register for dbGaP** (Framingham + MESA)
2. **Register for UK Biobank AMS**
3. **Contact Sax Institute** (45 and Up)
4. **Prepare IRB/HREC application**
5. **Draft research proposals** (2-3 pages each)

**Start applications within 1 week to have data in 3-4 months.**
