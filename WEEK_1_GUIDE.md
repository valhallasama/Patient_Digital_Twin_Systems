# Week 1: Getting Started - Action Guide

## 🎯 This Week's Goals

1. ✅ Register for CITI training
2. ✅ Complete "Data or Specimens Only Research" course (4-6 hours)
3. ✅ Create PhysioNet account
4. ✅ Set up reference manager (Zotero)
5. ✅ Create folder structure for literature review
6. ✅ Begin initial literature search

**Time commitment:** ~8-10 hours this week

---

## 📝 Task 1: CITI Training Registration (30 minutes)

### **Step-by-Step Instructions:**

1. **Go to CITI Program website**
   - URL: https://www.citiprogram.org/
   - Click "Register"

2. **Create Account**
   - Select "Create an Account"
   - Choose your institution (or "Not affiliated" if independent)
   - Fill in personal information
   - Create username and password

3. **Select Course**
   - After login, click "Add a Course"
   - Select: **"Data or Specimens Only Research"**
   - This is the required course for MIMIC-III access

4. **Course Modules (Complete All):**
   - [ ] Belmont Report and Its Principles
   - [ ] History and Ethical Principles - SBR
   - [ ] Defining Research with Human Subjects - SBR
   - [ ] Research with Protected Populations - SBR
   - [ ] Informed Consent - SBR
   - [ ] Privacy and Confidentiality - SBR
   - [ ] Records-Based Research
   - [ ] Conflicts of Interest in Research Involving Human Subjects

5. **Complete Course**
   - Each module has a quiz (must score 80%+)
   - Can retake quizzes if needed
   - Total time: 4-6 hours

6. **Download Certificate**
   - Once all modules complete, download completion certificate
   - Save as: `CITI_Certificate_[YourName].pdf`
   - You'll need this for PhysioNet application

**✅ Completion Criteria:** CITI certificate downloaded

---

## 📝 Task 2: PhysioNet Account Creation (15 minutes)

### **Step-by-Step Instructions:**

1. **Go to PhysioNet**
   - URL: https://physionet.org/
   - Click "Register" in top right

2. **Create Account**
   - Email address (use institutional email if possible)
   - Username
   - Password
   - First and last name
   - Affiliation (university/institution)

3. **Verify Email**
   - Check email for verification link
   - Click to activate account

4. **Complete Profile**
   - Add professional information
   - Research interests
   - Publications (if any)

**Note:** Don't apply for MIMIC-III access yet - wait until CITI training is complete

**✅ Completion Criteria:** PhysioNet account created and verified

---

## 📝 Task 3: Set Up Reference Manager (1 hour)

### **Recommended: Zotero (Free & Open Source)**

1. **Download Zotero**
   - URL: https://www.zotero.org/download/
   - Download Zotero Desktop App
   - Install browser connector (Chrome/Firefox)

2. **Create Account**
   - Sign up for free Zotero account
   - This syncs your library across devices

3. **Install Zotero**
   - Run installer
   - Sign in with your account

4. **Set Up Collections**
   - Create main collection: "Digital Twin Literature Review"
   - Create subcollections:
     - Metabolic System
     - Cardiovascular System
     - Renal System
     - Hepatic System
     - Immune System
     - Endocrine System
     - Neural System
     - Methods & Statistics

5. **Configure Settings**
   - Preferences → Cite → Styles
   - Install "Vancouver" style (medical journals)
   - Install "APA 7th" as backup

6. **Test Browser Connector**
   - Go to a PubMed article
   - Click Zotero connector in browser
   - Should automatically save citation

**Alternative:** Mendeley or EndNote (if you prefer)

**✅ Completion Criteria:** Zotero installed with collections created

---

## 📝 Task 4: Create Folder Structure (15 minutes)

### **Set Up Project Folders:**

```bash
cd ~/Yue/Patient_Digital_Twin_Systems

# Create literature review structure
mkdir -p literature_review/papers/{metabolic,cardiovascular,renal,hepatic,immune,endocrine,neural,methods}
mkdir -p literature_review/evidence_tables
mkdir -p literature_review/notes/weekly_summaries
mkdir -p literature_review/search_strategies

# Create data folder structure (for later)
mkdir -p data/{raw,processed,cohorts}
mkdir -p data/mimic-iii

# Create results folder
mkdir -p results/{models,figures,tables,reports}

# Create documentation
mkdir -p docs/{protocols,analysis_plans}
```

**Or manually create:**
```
Patient_Digital_Twin_Systems/
├── literature_review/
│   ├── papers/
│   │   ├── metabolic/
│   │   ├── cardiovascular/
│   │   ├── renal/
│   │   ├── hepatic/
│   │   ├── immune/
│   │   ├── endocrine/
│   │   ├── neural/
│   │   └── methods/
│   ├── evidence_tables/
│   ├── notes/
│   │   └── weekly_summaries/
│   └── search_strategies/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cohorts/
│   └── mimic-iii/
└── results/
    ├── models/
    ├── figures/
    ├── tables/
    └── reports/
```

**✅ Completion Criteria:** Folder structure created

---

## 📝 Task 5: Initial Literature Search (2-3 hours)

### **Start with High-Impact Papers:**

**Metabolic System - Top 10 Papers to Find:**

1. **UKPDS 16** (1995)
   - Search: "UKPDS 16 beta cell function"
   - PubMed ID: 7556954
   - Download PDF, save to `literature_review/papers/metabolic/`

2. **Diabetes Prevention Program** (NEJM 2002)
   - Search: "Diabetes Prevention Program NEJM 2002"
   - PMID: 11832527
   - Key paper for intervention effects

3. **ADOPT Trial** (Diabetes Care 2006)
   - Search: "ADOPT trial rosiglitazone metformin glyburide"
   - PMID: 17130203

4. **DeFronzo - Pathogenesis of Type 2 Diabetes** (Diabetes 2004)
   - Search: "DeFronzo pathogenesis type 2 diabetes 2004"
   - PMID: 15277396

5. **Kahn - Beta Cell Failure** (Diabetes 2006)
   - Search: "Kahn mechanisms obesity insulin resistance type 2 diabetes"
   - PMID: 16644707

**Search Strategy:**

1. **Go to PubMed:** https://pubmed.ncbi.nlm.nih.gov/

2. **Search Query:**
   ```
   ("beta cell function"[Title/Abstract] OR "insulin sensitivity"[Title/Abstract])
   AND ("decline"[Title/Abstract] OR "progression"[Title/Abstract])
   AND ("type 2 diabetes"[MeSH Terms])
   AND ("longitudinal"[Title/Abstract] OR "cohort"[Title/Abstract])
   ```

3. **Filter Results:**
   - Publication date: Last 20 years
   - Article type: Clinical Trial, Meta-Analysis, Systematic Review
   - Sort by: Best Match or Citation Count

4. **For Each Paper:**
   - Click Zotero connector to save citation
   - Download PDF (click "Full Text Links")
   - Save PDF to appropriate folder
   - Add notes in Zotero about relevance

5. **Create Search Log:**
   - Document search terms used
   - Number of results
   - Papers selected
   - Save in `literature_review/search_strategies/metabolic_search_log.txt`

**✅ Completion Criteria:** 
- Found and saved 5-10 key metabolic papers
- Citations in Zotero
- PDFs downloaded and organized

---

## 📝 Task 6: Create Evidence Table Template (30 minutes)

### **Set Up Excel Template:**

Create file: `literature_review/evidence_tables/Metabolic_Parameters_Template.xlsx`

**Columns:**
1. Paper_ID (e.g., "UKPDS_16_1995")
2. First_Author
3. Year
4. Title
5. Journal
6. PMID
7. DOI
8. Study_Design (RCT/Cohort/Cross-sectional/Meta-analysis)
9. Sample_Size
10. Population_Description
11. Follow_up_Duration
12. Parameter_Name
13. Parameter_Value
14. Parameter_Unit
15. Standard_Deviation
16. Confidence_Interval_95
17. Subgroup (if applicable)
18. Quality_Assessment (High/Medium/Low)
19. Relevance (High/Medium/Low)
20. Notes
21. Extraction_Date
22. Extracted_By

**Create template with example row:**
```
Paper_ID: UKPDS_16_1995
First_Author: UK Prospective Diabetes Study Group
Year: 1995
Title: U.K. Prospective Diabetes Study 16: Overview of 6 years' therapy...
Journal: Diabetes
PMID: 7556954
Study_Design: RCT
Sample_Size: 3867
Population: Newly diagnosed type 2 diabetes
Follow_up: 6 years
Parameter_Name: Beta cell function decline
Parameter_Value: 4
Parameter_Unit: % per year
Standard_Deviation: 2.1
Confidence_Interval: 3.2-4.8
Quality: High
Relevance: High
Notes: Landmark study, large sample, long follow-up
```

**✅ Completion Criteria:** Evidence table template created

---

## 📊 Week 1 Progress Tracker

### **Daily Checklist:**

**Monday:**
- [ ] Register for CITI training
- [ ] Start first module (Belmont Report)

**Tuesday:**
- [ ] Complete 2-3 CITI modules
- [ ] Create PhysioNet account

**Wednesday:**
- [ ] Complete remaining CITI modules
- [ ] Download certificate

**Thursday:**
- [ ] Install Zotero
- [ ] Set up folder structure
- [ ] Begin literature search

**Friday:**
- [ ] Find and save 5-10 key papers
- [ ] Create evidence table template
- [ ] Document search strategies

**Weekend:**
- [ ] Review week's progress
- [ ] Plan Week 2 activities
- [ ] Optional: Read 1-2 papers in detail

---

## 📝 End of Week 1 Deliverables

**You should have:**
1. ✅ CITI completion certificate
2. ✅ PhysioNet account (verified)
3. ✅ Zotero installed with collections
4. ✅ Folder structure created
5. ✅ 5-10 papers found and saved
6. ✅ Evidence table template
7. ✅ Search strategy documented

**Time spent:** ~8-10 hours

---

## 🚀 Week 2 Preview

**Next week you'll:**
- Apply for MIMIC-III access (using CITI certificate)
- Continue literature review (aim for 10-15 more papers)
- Start extracting parameters from papers
- Begin filling evidence table

---

## 💡 Tips for Success

1. **Block Time:** Schedule 1-2 hour blocks for focused work
2. **CITI Training:** Do in one sitting if possible (4-6 hours)
3. **Literature Search:** Start broad, then narrow down
4. **Organization:** Keep everything organized from day 1
5. **Notes:** Document everything - you'll thank yourself later
6. **Ask Questions:** If stuck, reach out for help

---

## 📞 Resources

**CITI Training:**
- Website: https://www.citiprogram.org/
- Help: https://support.citiprogram.org/

**PhysioNet:**
- Website: https://physionet.org/
- MIMIC-III docs: https://mimic.mit.edu/docs/

**Zotero:**
- Download: https://www.zotero.org/download/
- Documentation: https://www.zotero.org/support/

**PubMed:**
- Search: https://pubmed.ncbi.nlm.nih.gov/
- Advanced search: https://pubmed.ncbi.nlm.nih.gov/advanced/

---

## ✅ Week 1 Complete!

Once you've completed all tasks, you're ready for Week 2!

**Next steps:**
1. Apply for MIMIC-III access
2. Continue literature review
3. Start parameter extraction

**Good luck! 🚀**
