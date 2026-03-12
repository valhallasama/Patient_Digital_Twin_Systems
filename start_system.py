#!/usr/bin/env python3
"""
Patient Digital Twin System - Interactive Demo
Start the complete system for testing
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from digital_twin_system import DigitalTwinSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print system banner"""
    print("\n" + "="*80)
    print("PATIENT DIGITAL TWIN SYSTEM - INTERACTIVE DEMO")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Multi-agent medical analysis (5 specialist agents)")
    print("  ✓ Disease progression simulation")
    print("  ✓ ML risk prediction (trained on 102K+ real patients)")
    print("  ✓ Intervention simulation")
    print("  ✓ Temporal health tracking")
    print("  ✓ Knowledge graph integration")
    print("\n" + "="*80 + "\n")


def create_sample_medical_report():
    """Create a sample medical report for testing"""
    return """
PATIENT MEDICAL REPORT

Patient ID: TEST-001
Date: 2024-03-12
Age: 58 years
Gender: Male

CHIEF COMPLAINT:
Patient presents with chest pain and shortness of breath for the past 2 weeks.

VITAL SIGNS:
- Blood Pressure: 165/95 mmHg (elevated)
- Heart Rate: 92 bpm
- Temperature: 98.6°F
- Respiratory Rate: 18/min
- BMI: 31.2 (obese)

MEDICAL HISTORY:
- Type 2 Diabetes Mellitus (diagnosed 5 years ago)
- Hypertension (diagnosed 3 years ago)
- Hyperlipidemia
- Former smoker (quit 2 years ago, 20 pack-years)
- Family history of coronary artery disease (father had MI at age 62)

CURRENT MEDICATIONS:
- Metformin 1000mg BID
- Lisinopril 20mg daily
- Atorvastatin 40mg daily
- Aspirin 81mg daily

LABORATORY RESULTS:
- HbA1c: 8.2% (elevated, poor glycemic control)
- Fasting Glucose: 165 mg/dL (elevated)
- Total Cholesterol: 240 mg/dL (elevated)
- LDL: 160 mg/dL (elevated)
- HDL: 38 mg/dL (low)
- Triglycerides: 210 mg/dL (elevated)
- Creatinine: 1.3 mg/dL (slightly elevated)
- eGFR: 58 mL/min/1.73m² (Stage 3 CKD)

PHYSICAL EXAMINATION:
- General: Alert and oriented, appears anxious
- Cardiovascular: Regular rhythm, no murmurs, S4 gallop present
- Respiratory: Clear to auscultation bilaterally
- Extremities: Trace pedal edema bilaterally
- Neurological: Intact, no focal deficits

ECG FINDINGS:
- Sinus rhythm
- Left ventricular hypertrophy
- Non-specific ST-T wave changes in lateral leads

ASSESSMENT:
1. Chest pain - rule out acute coronary syndrome
2. Uncontrolled Type 2 Diabetes Mellitus
3. Hypertension, inadequately controlled
4. Dyslipidemia
5. Chronic Kidney Disease Stage 3
6. Obesity

PLAN:
- Admit for cardiac workup
- Troponin levels, stress test
- Optimize diabetes management
- Adjust antihypertensive regimen
- Lifestyle modifications counseling
- Nephrology consult for CKD management
"""


def run_demo():
    """Run interactive demo"""
    print_banner()
    
    print("Initializing Patient Digital Twin System...")
    print("Loading ML models trained on 102,363 real patients...\n")
    
    # Initialize system
    system = DigitalTwinSystem()
    
    print("✓ System initialized successfully!\n")
    print("="*80)
    print("DEMO: ANALYZING SAMPLE PATIENT")
    print("="*80)
    
    # Get sample report
    medical_report = create_sample_medical_report()
    
    print("\nMedical Report:")
    print("-" * 80)
    print(medical_report[:500] + "...\n")
    
    print("="*80)
    print("STARTING ANALYSIS")
    print("="*80)
    print("\nThis will:")
    print("  1. Extract patient information")
    print("  2. Run 5 specialist agents (Cardiologist, Endocrinologist, etc.)")
    print("  3. Simulate disease progression (5 years)")
    print("  4. Calculate ML risk predictions")
    print("  5. Simulate interventions")
    print("  6. Generate comprehensive report")
    print("\nEstimated time: 30-60 seconds\n")
    
    input("Press ENTER to start analysis...")
    
    print("\n" + "="*80)
    print("RUNNING ANALYSIS...")
    print("="*80 + "\n")
    
    # Run analysis
    try:
        result = system.analyze_patient(
            medical_report=medical_report,
            patient_id="TEST-001",
            simulation_years=5
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        # Display results
        if result.get('success'):
            print("\n✓ Analysis completed successfully")
            print(f"\nPatient ID: {result.get('patient_id')}")
            print(f"Analysis timestamp: {result.get('timestamp')}")
            
            # Show key findings
            if 'summary' in result:
                print("\n" + "-"*80)
                print("KEY FINDINGS:")
                print("-"*80)
                print(result['summary'][:1000])
            
            # Show risk predictions
            if 'risk_predictions' in result:
                print("\n" + "-"*80)
                print("ML RISK PREDICTIONS (from 102K real patients):")
                print("-"*80)
                for disease, risk in result['risk_predictions'].items():
                    print(f"  {disease}: {risk:.1%}")
            
            # Show simulation results
            if 'disease_progression' in result:
                print("\n" + "-"*80)
                print("5-YEAR DISEASE PROGRESSION SIMULATION:")
                print("-"*80)
                prog = result['disease_progression']
                print(f"  Baseline HbA1c: {prog.get('baseline_hba1c', 'N/A')}")
                print(f"  Projected HbA1c (5yr): {prog.get('projected_hba1c', 'N/A')}")
                print(f"  Cardiovascular risk increase: {prog.get('cv_risk_increase', 'N/A')}")
            
            # Save report
            report_path = Path("outputs") / f"patient_report_{result.get('patient_id')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            import json
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\n✓ Full report saved: {report_path}")
            
        else:
            print("\n✗ Analysis failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        print("\nNote: Some components may need API keys (OpenAI/Anthropic)")
        print("The system will use fallback methods where possible.")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the generated report")
    print("  2. Try with your own medical reports")
    print("  3. Explore individual components:")
    print("     - Multi-agent analysis: agents/")
    print("     - ML models: models/real_data/")
    print("     - Disease simulation: simulation_engine/")
    print("\n")


def main():
    """Main entry point"""
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
