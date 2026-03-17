# 🧹 Project Cleanup Plan

## Files to Remove (Obsolete/Redundant)

### **Documentation (46 MD files → Keep 5 essential)**

**KEEP (Essential Documentation):**
1. `README.md` - Main project documentation
2. `RESEARCH_ARCHITECTURE.md` - Complete architecture (NEW)
3. `HYBRID_SYSTEM_COMPLETE.md` - Hybrid AI explanation
4. `MULTI_PARAMETER_EVOLUTION_COMPLETE.md` - Parameter simulation
5. `QUICK_START.md` - Getting started guide

**REMOVE (Redundant/Obsolete):**
- `ARCHITECTURE_UPGRADE_PLAN.md` - Superseded by RESEARCH_ARCHITECTURE.md
- `BACKUP_INFO.md` - Not needed
- `CLARIFICATION_GPT_VS_RULE_BASED.md` - Covered in HYBRID_SYSTEM_COMPLETE.md
- `COMPLETE_SYSTEM_GUIDE.md` - Redundant with README + RESEARCH_ARCHITECTURE
- `COMPREHENSIVE_SYSTEM_DESIGN.md` - Superseded
- `CRITICAL_ISSUES_ANALYSIS.md` - Historical, no longer relevant
- `CURRENT_STATUS_SUMMARY.md` - Outdated
- `DATASET_ACQUISITION_SUMMARY.md` - Historical
- `DATA_ANALYSIS_PIPELINE.md` - Not core functionality
- `DATA_GENERATION_EXPLAINED.md` - Historical
- `FINAL_DATASET_STATUS.md` - Historical
- `FUNCTIONAL_PROTOTYPE_COMPLETE.md` - Historical milestone
- `GPT_FREE_ALTERNATIVES.md` - Not relevant (we use hybrid approach)
- `HYBRID_MODEL_SUMMARY.md` - Covered in HYBRID_SYSTEM_COMPLETE.md
- `INTEGRATION_GUIDE.md` - Redundant
- `MIROFISH_IMPLEMENTATION_GUIDE.md` - Superseded by RESEARCH_ARCHITECTURE
- `MIROFISH_LEVEL_ASSESSMENT.md` - Historical
- `MIROFISH_RESEARCH_AND_IMPLEMENTATION.md` - Superseded
- `MIROFISH_TRANSFORMATION_PLAN.md` - Completed, now in RESEARCH_ARCHITECTURE
- `NEXT_STEPS.md` - Outdated
- `PARAMETER_AUDIT.md` - Historical
- `PREDICTION_SYSTEM_ANALYSIS.md` - Historical
- `PROJECT_SUMMARY.md` - Covered in README
- `QUICK_START_GUIDE.md` - Duplicate of QUICK_START.md
- `QWEN_VS_RESEARCH.md` - Not relevant
- `README_REAL_DATA.md` - Merge into README
- `REAL_DATA_ACQUISITION_GUIDE.md` - Historical
- `REAL_DATA_COMPLETE.md` - Historical
- `REAL_DATA_SOURCES.md` - Historical
- `REAL_DATA_STATUS.md` - Historical
- `REALITY_CHECK.md` - Historical
- `REPORT_UPLOAD_COMPLETE.md` - Historical
- `RESEARCH_PLAN.md` - Completed, now in RESEARCH_ARCHITECTURE
- `SIMULATION_BASED_INTERVENTIONS_GUIDE.md` - Now in scenario_simulator.py
- `START_HERE.md` - Redundant with QUICK_START
- `SYSTEM_ARCHITECTURE_V2.md` - Superseded by RESEARCH_ARCHITECTURE
- `SYSTEM_STATUS.md` - Outdated
- `TEMPORAL_SIMULATION_COMPLETE.md` - Keep (explains temporal evolution)
- `TEST_RESULTS_SUMMARY.md` - Keep (validation results)
- `WEEK_1_GUIDE.md` - Historical
- `WHY_NO_DISEASES_AND_SOLUTIONS.md` - Historical

**Total to remove: 41 MD files**

---

### **Python Scripts (30+ files → Keep 10 essential)**

**KEEP (Core Functionality):**
1. `test_all_parameters.py` - Parameter evolution tests
2. `test_temporal_simulation.py` - Temporal simulation tests
3. `train_comprehensive_models.py` - ML model training
4. `start_system.py` - System launcher (if used)

**REMOVE (Obsolete Demos):**
- `demo_complete_system.py` - Superseded by web_app
- `demo_comprehensive_twin.py` - Superseded by web_app
- `demo_gpt_free.py` - Not relevant
- `demo_mirofish_patient.py` - Superseded by web_app
- `demo_mirofish_with_llm.py` - Superseded by web_app
- `demo_simulation_based_interventions.py` - Now in scenario_simulator.py
- `demo_with_lifestyle.py` - Superseded by lifestyle_agent.py
- `run_demo.py` - Superseded by web_app

**REMOVE (Redundant Data Scripts):**
- `analyze_generated_data.py` - Not core
- `compare_real_vs_synthetic.py` - Not core
- `download_additional_datasets.py` - Not core
- `download_all_medical_datasets.py` - Not core
- `download_awesome_datasets.py` - Not core
- `download_real_datasets.py` - Not core
- `generate_data_auto.py` - Not core
- `generate_massive_data.py` - Not core
- `integrate_all_real_data.py` - Not core
- `integrate_real_data_now.py` - Not core
- `run_daily_data_acquisition.py` - Not core
- `web_dataset_scraper.py` - Not core

**REMOVE (Redundant Training Scripts):**
- `train_all_real_datasets.py` - Superseded by train_comprehensive_models.py
- `train_hybrid_model.py` - Superseded by train_comprehensive_models.py
- `train_ml_models_full.py` - Superseded by train_comprehensive_models.py
- `train_ml_models.py` - Superseded by train_comprehensive_models.py
- `train_ml_models_real.py` - Superseded by train_comprehensive_models.py
- `train_on_real_data_now.py` - Superseded by train_comprehensive_models.py

**REMOVE (Redundant System Scripts):**
- `digital_twin_system.py` - Superseded by mirofish_engine/
- `research_analysis.py` - Not core
- `test_system.py` - Superseded by test_all_parameters.py
- `web_app.py` - Superseded by web_app/app.py

**Total to remove: ~26 Python files**

---

### **Directories to Clean**

**KEEP:**
- `mirofish_engine/` - Core simulation engine ✅
- `patient_state/` - State model ✅
- `llm_integration/` - LLM layer ✅
- `simulation_engine/` - Scenario simulator ✅
- `web_app/` - UI (MUST KEEP) ✅
- `models/` - ML models ✅
- `data/` - Datasets (keep structure, can clean contents)
- `tests/` - If has useful tests

**REVIEW/CLEAN:**
- `agents/` - Check if superseded by mirofish_engine/comprehensive_agents.py
- `ai_core/` - Check if superseded
- `api/` - Check if superseded by web_app/
- `core/` - Check if superseded
- `dashboard/` - Check if superseded by web_app/
- `data_cleaning/` - Likely not needed
- `data_engine/` - Check if superseded
- `data_pipeline/` - Check if superseded
- `database/` - Check if used
- `knowledge_graph/` - Check if used
- `literature_review/` - Not core
- `prediction_engine/` - Check if superseded by mirofish_engine/
- `synthetic_data_generator/` - Not core
- `utils/` - Keep if has useful utilities

---

## Cleanup Strategy

1. **Move obsolete docs to archive/** (don't delete immediately)
2. **Remove redundant Python scripts**
3. **Clean up duplicate directories**
4. **Keep web_app/ fully functional**
5. **Update README.md with new structure**
6. **Test web UI after cleanup**

---

## Final Structure (Clean)

```
Patient_Digital_Twin_Systems/
├── README.md
├── QUICK_START.md
├── RESEARCH_ARCHITECTURE.md
├── HYBRID_SYSTEM_COMPLETE.md
├── MULTI_PARAMETER_EVOLUTION_COMPLETE.md
├── TEMPORAL_SIMULATION_COMPLETE.md
├── TEST_RESULTS_SUMMARY.md
│
├── mirofish_engine/          # Core simulation
├── patient_state/            # State model
├── llm_integration/          # LLM layer
├── simulation_engine/        # Scenarios
├── web_app/                  # UI (KEEP)
├── models/                   # ML models
├── data/                     # Datasets
├── utils/                    # Utilities
│
├── test_all_parameters.py
├── test_temporal_simulation.py
├── train_comprehensive_models.py
└── requirements.txt
```

**Result: Clean, focused, research-grade structure**
