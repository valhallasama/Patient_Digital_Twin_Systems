# Literature Review Tracking

## 📚 Overview

Systematic review of 200+ papers to extract physiological parameters for digital twin model.

**Timeline:** Weeks 2-14 (12 weeks)  
**Target:** 200-300 papers  
**Output:** Evidence table with all parameters and citations

---

## 📋 Progress Tracking

### **Metabolic System (50 papers)**

**Status:** Not started  
**Timeline:** Weeks 2-4

| Paper | Status | Parameters Extracted | Notes |
|-------|--------|---------------------|-------|
| UKPDS 16 (Diabetes 1995) | ⬜ Pending | Beta cell decline rate | - |
| ADOPT trial (Diabetes Care 2006) | ⬜ Pending | Glycemic durability | - |
| Kahn SE (Diabetes 2006) | ⬜ Pending | Beta cell failure mechanisms | - |
| ... | | | |

### **Cardiovascular System (50 papers)**

**Status:** Not started  
**Timeline:** Weeks 5-7

| Paper | Status | Parameters Extracted | Notes |
|-------|--------|---------------------|-------|
| MESA study | ⬜ Pending | IMT progression, CAC | - |
| Framingham Offspring | ⬜ Pending | CVD risk factors | - |
| ... | | | |

### **Renal System (50 papers)**

**Status:** Not started  
**Timeline:** Weeks 8-10

| Paper | Status | Parameters Extracted | Notes |
|-------|--------|---------------------|-------|
| KDIGO 2012 | ⬜ Pending | CKD classification, decline rates | - |
| Coresh et al. (JASN 2014) | ⬜ Pending | Population eGFR decline | - |
| ... | | | |

### **Other Systems (50 papers)**

**Status:** Not started  
**Timeline:** Weeks 11-13

---

## 📊 Evidence Table Template

For each paper, extract:

```
Paper ID: [First author] [Year]
Title: [Full title]
Journal: [Journal name]
PMID: [PubMed ID]
DOI: [DOI]

Study Design: [RCT/Cohort/Cross-sectional/Meta-analysis]
Sample Size: [N]
Population: [Description]
Follow-up: [Duration]

Parameters Extracted:
- Parameter 1: [Name]
  - Value: [Mean ± SD]
  - Unit: [Unit]
  - Subgroups: [If applicable]
  - Confidence Interval: [95% CI]
  
- Parameter 2: ...

Quality Assessment: [High/Medium/Low]
Relevance: [High/Medium/Low]
Notes: [Any important details]
```

---

## 🔍 Search Strategy

### **Databases:**
- PubMed
- Google Scholar
- Cochrane Library
- Web of Science

### **Search Terms:**

**Metabolic:**
```
("beta cell function" OR "insulin sensitivity" OR "HOMA-IR") 
AND ("decline" OR "progression" OR "longitudinal")
AND ("type 2 diabetes" OR "prediabetes")
```

**Cardiovascular:**
```
("atherosclerosis" OR "carotid IMT" OR "arterial stiffness")
AND ("progression" OR "change")
AND ("longitudinal" OR "cohort")
```

**Renal:**
```
("eGFR" OR "kidney function" OR "CKD")
AND ("decline" OR "progression")
AND ("diabetes" OR "hypertension")
```

---

## 📁 File Organization

```
literature_review/
├── README.md (this file)
├── papers/
│   ├── metabolic/
│   │   ├── UKPDS_16_1995.pdf
│   │   ├── ADOPT_2006.pdf
│   │   └── ...
│   ├── cardiovascular/
│   ├── renal/
│   └── other/
├── evidence_tables/
│   ├── Metabolic_Parameters.xlsx
│   ├── Cardiovascular_Parameters.xlsx
│   ├── Renal_Parameters.xlsx
│   └── Master_Evidence_Table.xlsx
├── notes/
│   └── weekly_summaries/
└── references.bib
```

---

## 🎯 Next Steps

1. **Set up reference manager** (Zotero recommended)
2. **Create folder structure** as above
3. **Start with high-impact papers** (UKPDS, MESA, KDIGO)
4. **Extract parameters systematically**
5. **Update progress weekly**

---

## 📞 Resources

- **Zotero:** https://www.zotero.org/
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/
- **Google Scholar:** https://scholar.google.com/
- **PRISMA guidelines:** http://www.prisma-statement.org/
