#!/usr/bin/env python3
"""
Comprehensive Patient Profile Examples

Demonstrates the full organ coverage of the digital twin system.
Shows how the system handles:
1. Complete patient data (all parameters provided)
2. Partial patient data (missing parameters inferred)
3. Minimal patient data (maximum inference)
"""

from parameter_inference import ParameterInferenceEngine


def create_comprehensive_patient_complete():
    """
    Example: Complete patient data for ALL organ systems
    
    This is the ideal case - all biomarkers available
    """
    return {
        'patient_id': 'patient_complete_001',
        'demographics': {
            'age': 52,
            'gender': 'female',
            'bmi': 31.2,
            'height_cm': 165,
            'weight_kg': 85,
            'ethnicity': 'caucasian'
        },
        'organ_biomarkers': {
            'metabolic': {
                'glucose': 118,           # mg/dL - Pre-diabetic
                'HbA1c': 6.2,            # % - Pre-diabetic
                'insulin': 22,            # μU/mL - Insulin resistance
                'triglycerides': 195      # mg/dL - Borderline high
            },
            'cardiovascular': {
                'systolic_bp': 142,       # mmHg - Stage 1 hypertension
                'diastolic_bp': 88,       # mmHg - Borderline
                'total_cholesterol': 235, # mg/dL - Borderline high
                'HDL': 42,                # mg/dL - Low (risk factor)
                'LDL': 155                # mg/dL - Borderline high
            },
            'liver': {
                'ALT': 58,                # U/L - Elevated (fatty liver)
                'AST': 42                 # U/L - Mildly elevated
            },
            'kidney': {
                'creatinine': 0.95,       # mg/dL - Normal
                'BUN': 18                 # mg/dL - Normal
            },
            'immune': {
                'WBC': 8.2                # K/μL - Normal
            },
            'neural': {
                'cognitive_score': 0.82   # Normalized - Mild decline
            },
            'lifestyle': {
                'exercise_frequency': 0.15,    # Sedentary
                'alcohol_consumption': 0.6,    # Moderate-heavy
                'diet_quality': 0.25,          # Poor
                'sleep_hours': 6.0,            # Insufficient
                'smoking': 0.0                 # Non-smoker
            }
        },
        'medical_history': [
            'Pre-diabetes diagnosed 2 years ago',
            'Fatty liver detected on ultrasound',
            'Family history of type 2 diabetes (mother)',
            'Menopause at age 50'
        ],
        'medications': [
            'Metformin 500mg (for pre-diabetes)',
            'Multivitamin'
        ],
        'lifestyle_notes': 'Desk job, minimal physical activity, stress eating, wine with dinner daily'
    }


def create_comprehensive_patient_partial():
    """
    Example: Partial patient data - common lab panel
    
    Typical scenario: Patient has recent lab work but not complete panel
    Missing: Insulin, some cholesterol, neural assessment, lifestyle details
    """
    return {
        'patient_id': 'patient_partial_002',
        'demographics': {
            'age': 38,
            'gender': 'male',
            'bmi': 27.8,
            'height_cm': 178,
            'weight_kg': 88
        },
        'organ_biomarkers': {
            'metabolic': {
                'glucose': 102,           # mg/dL - Provided
                'HbA1c': 5.8,            # % - Provided
                # insulin: MISSING - will be inferred
                'triglycerides': 165      # mg/dL - Provided
            },
            'cardiovascular': {
                'systolic_bp': 138,       # mmHg - Provided
                'diastolic_bp': 86,       # mmHg - Provided
                'total_cholesterol': 215, # mg/dL - Provided
                # HDL: MISSING - will be inferred
                # LDL: MISSING - will be calculated
            },
            'liver': {
                'ALT': 48                 # U/L - Provided
                # AST: MISSING - will be inferred
            },
            'kidney': {
                'creatinine': 1.05        # mg/dL - Provided
                # BUN: MISSING - will be inferred
            }
            # immune: MISSING - will be inferred
            # neural: MISSING - will be inferred
            # lifestyle: MISSING - will be reverse-inferred from biomarkers
        },
        'medical_history': [
            'No chronic conditions',
            'Occasional headaches'
        ],
        'medications': []
    }


def create_comprehensive_patient_minimal():
    """
    Example: Minimal patient data - only basic vitals
    
    Extreme scenario: Patient only has BP and glucose from pharmacy screening
    Everything else must be inferred
    """
    return {
        'patient_id': 'patient_minimal_003',
        'demographics': {
            'age': 45,
            'gender': 'male',
            'bmi': 29.5  # Calculated from height/weight if available
        },
        'organ_biomarkers': {
            'metabolic': {
                'glucose': 115            # mg/dL - Only parameter provided
            },
            'cardiovascular': {
                'systolic_bp': 145        # mmHg - Only parameter provided
            }
            # Everything else MISSING - will be inferred
        },
        'medical_history': [],
        'medications': []
    }


def create_athlete_profile():
    """
    Example: Healthy athlete profile
    
    Shows what optimal biomarkers look like
    """
    return {
        'patient_id': 'athlete_001',
        'demographics': {
            'age': 32,
            'gender': 'male',
            'bmi': 22.5,
            'height_cm': 180,
            'weight_kg': 73
        },
        'organ_biomarkers': {
            'metabolic': {
                'glucose': 88,
                'HbA1c': 5.1,
                'insulin': 8,
                'triglycerides': 75
            },
            'cardiovascular': {
                'systolic_bp': 112,
                'diastolic_bp': 72,
                'total_cholesterol': 165,
                'HDL': 62,
                'LDL': 88
            },
            'liver': {
                'ALT': 22,
                'AST': 19
            },
            'kidney': {
                'creatinine': 1.1,  # Higher due to muscle mass
                'BUN': 14
            },
            'immune': {
                'WBC': 6.5
            },
            'neural': {
                'cognitive_score': 0.95
            },
            'lifestyle': {
                'exercise_frequency': 0.9,  # Daily training
                'alcohol_consumption': 0.1,  # Minimal
                'diet_quality': 0.85,        # Excellent
                'sleep_hours': 8.5,
                'smoking': 0.0
            }
        },
        'medical_history': ['No chronic conditions'],
        'medications': [],
        'lifestyle_notes': 'Marathon runner, plant-based diet, stress management practices'
    }


def create_metabolic_syndrome_profile():
    """
    Example: Metabolic syndrome patient
    
    Shows clustering of cardiovascular risk factors
    """
    return {
        'patient_id': 'metabolic_syndrome_001',
        'demographics': {
            'age': 58,
            'gender': 'male',
            'bmi': 34.5,
            'height_cm': 175,
            'weight_kg': 106
        },
        'organ_biomarkers': {
            'metabolic': {
                'glucose': 135,           # Diabetic range
                'HbA1c': 7.2,            # Diabetic
                'insulin': 35,            # Severe insulin resistance
                'triglycerides': 285      # Very high
            },
            'cardiovascular': {
                'systolic_bp': 155,       # Stage 2 hypertension
                'diastolic_bp': 95,       # Stage 2 hypertension
                'total_cholesterol': 245,
                'HDL': 35,                # Very low (high risk)
                'LDL': 168                # High
            },
            'liver': {
                'ALT': 85,                # Elevated (NAFLD)
                'AST': 62
            },
            'kidney': {
                'creatinine': 1.3,        # Mildly elevated
                'BUN': 22
            },
            'immune': {
                'WBC': 9.5                # Elevated (inflammation)
            },
            'neural': {
                'cognitive_score': 0.75   # Vascular cognitive impairment
            },
            'lifestyle': {
                'exercise_frequency': 0.05,
                'alcohol_consumption': 0.4,
                'diet_quality': 0.2,
                'sleep_hours': 5.5,
                'smoking': 0.0
            }
        },
        'medical_history': [
            'Type 2 diabetes diagnosed 3 years ago',
            'Hypertension diagnosed 5 years ago',
            'Non-alcoholic fatty liver disease',
            'Obstructive sleep apnea',
            'Family history: father had MI at age 55'
        ],
        'medications': [
            'Metformin 1000mg BID',
            'Lisinopril 20mg daily',
            'Atorvastatin 40mg daily',
            'Aspirin 81mg daily'
        ]
    }


def demonstrate_all_profiles():
    """Demonstrate inference for all patient types"""
    engine = ParameterInferenceEngine()
    
    profiles = [
        ("Complete Data", create_comprehensive_patient_complete()),
        ("Partial Data", create_comprehensive_patient_partial()),
        ("Minimal Data", create_comprehensive_patient_minimal()),
        ("Athlete", create_athlete_profile()),
        ("Metabolic Syndrome", create_metabolic_syndrome_profile())
    ]
    
    for name, patient in profiles:
        print("\n" + "="*80)
        print(f"PATIENT PROFILE: {name}")
        print("="*80)
        print(f"ID: {patient['patient_id']}")
        print(f"Demographics: Age={patient['demographics']['age']}, "
              f"Gender={patient['demographics']['gender']}, "
              f"BMI={patient['demographics']['bmi']:.1f}")
        
        # Count provided vs missing parameters
        provided_count = 0
        total_params = 0
        
        for organ, biomarkers in patient.get('organ_biomarkers', {}).items():
            for param in biomarkers:
                provided_count += 1
                total_params += 1
        
        # Expected total parameters
        expected_total = 19  # From ORGAN_PARAMETERS_COMPLETE.md
        missing_count = expected_total - provided_count
        
        print(f"\nData Completeness: {provided_count}/{expected_total} parameters provided "
              f"({provided_count/expected_total*100:.0f}%)")
        
        if missing_count > 0:
            print(f"Missing {missing_count} parameters - will be inferred")
            
            # Perform inference
            complete_profile, inference_notes = engine.infer_complete_profile(
                patient.get('organ_biomarkers', {}),
                patient['demographics']
            )
            
            print("\nInferred Parameters:")
            for key, note in inference_notes.items():
                if "Inferred" in note or "Reverse-inferred" in note or "Calculated" in note:
                    organ, param = key.split('.')
                    value = complete_profile[organ][param]
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f} - {note}")
                    else:
                        print(f"  {key}: {value} - {note}")
        else:
            print("Complete data - no inference needed")
        
        # Show medical context
        if patient.get('medical_history'):
            print(f"\nMedical History: {'; '.join(patient['medical_history'][:2])}")
        if patient.get('medications'):
            print(f"Medications: {'; '.join(patient['medications'][:2])}")


if __name__ == '__main__':
    demonstrate_all_profiles()
