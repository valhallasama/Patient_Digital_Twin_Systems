"""
NHANES Variable Mapping System

Maps harmonized NHANES CSV variable names to standard features needed for digital twin system.
Based on the NHANES 1988-2018 harmonized dataset structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class NHANESVariableMapper:
    """Maps harmonized NHANES CSV variables to standard feature names."""
    
    def __init__(self):
        self.demographics_map = self._create_demographics_map()
        self.lab_map = self._create_lab_map()
        self.questionnaire_map = self._create_questionnaire_map()
        self.derived_calculations = self._create_derived_calculations()
    
    def _create_demographics_map(self) -> Dict[str, str]:
        """Map demographics variables from CSV to standard names."""
        return {
            'SEQN': 'patient_id',
            'RIDAGEYR': 'age',
            'RIAGENDR': 'sex',
            'RIDRETH1': 'race_ethnicity',
            'RIDRETH3': 'race_ethnicity_detailed',
            'DMDEDUC2': 'education',
            'INDFMPIR': 'poverty_income_ratio',
            'RIDEXPRG': 'pregnancy_status',
        }
    
    def _create_lab_map(self) -> Dict[str, str]:
        """Map laboratory/chemical variables to standard names."""
        return {
            'LBXGLU': 'glucose',
            'LBXGH': 'hba1c',
            'LBXGHC': 'hba1c',
            'LBXGHCLA': 'hba1c',
            
            'BPXSY1': 'systolic_bp',
            'BPXSY2': 'systolic_bp_2',
            'BPXSY3': 'systolic_bp_3',
            'BPXDI1': 'diastolic_bp',
            'BPXDI2': 'diastolic_bp_2',
            'BPXDI3': 'diastolic_bp_3',
            
            'BMXBMI': 'bmi',
            'BMXWT': 'weight_kg',
            'BMXHT': 'height_cm',
            'BMXWAIST': 'waist_circumference',
            
            'LBXTC': 'total_cholesterol',
            'LBXTR': 'triglycerides',
            'LBDHDD': 'hdl',
            'LBDLDL': 'ldl',
            
            'LBXSCR': 'creatinine',
            'LBXSUA': 'uric_acid',
            'LBXSBU': 'bun',
            'LBXSAL': 'albumin',
            
            'LBXSASSI': 'ast',
            'LBXSATSI': 'alt',
            'LBXSAPSI': 'alkaline_phosphatase',
            'LBXSGTSI': 'ggt',
            
            'LBXWBCSI': 'wbc',
            'LBXRBCSI': 'rbc',
            'LBXHGB': 'hemoglobin',
            'LBXHCT': 'hematocrit',
            'LBXPLTSI': 'platelets',
            
            'LBXSTP': 'total_protein',
            'LBXSCA': 'calcium',
            'LBXSPH': 'phosphorus',
            'LBXSNASI': 'sodium',
            'LBXSKSI': 'potassium',
            'LBXSCLSI': 'chloride',
            
            'URXUMA': 'urine_albumin',
            'URXUCR': 'urine_creatinine',
        }
    
    def _create_questionnaire_map(self) -> Dict[str, str]:
        """Map questionnaire variables to standard names."""
        return {
            'SMQ020': 'ever_smoked',
            'SMQ040': 'current_smoker',
            'SMD030': 'age_started_smoking',
            'SMD650': 'cigarettes_per_day',
            
            'ALQ101': 'ever_alcohol',
            'ALQ120Q': 'alcohol_frequency',
            'ALQ130': 'drinks_per_day',
            
            'PAQ605': 'vigorous_work',
            'PAQ620': 'moderate_work',
            'PAQ635': 'walk_bicycle',
            'PAQ650': 'vigorous_recreation',
            'PAQ665': 'moderate_recreation',
            
            'DIQ010': 'doctor_told_diabetes',
            'DIQ050': 'taking_insulin',
            'DIQ070': 'taking_diabetes_pills',
            
            'BPQ020': 'doctor_told_hypertension',
            'BPQ040A': 'taking_bp_medication',
            
            'MCQ160B': 'doctor_told_chf',
            'MCQ160C': 'doctor_told_chd',
            'MCQ160D': 'doctor_told_angina',
            'MCQ160E': 'doctor_told_heart_attack',
            'MCQ160F': 'doctor_told_stroke',
            
            'MCQ220': 'doctor_told_cancer',
            'KIQ022': 'doctor_told_kidney_disease',
            'MCQ160L': 'doctor_told_liver_condition',
        }
    
    def _create_derived_calculations(self) -> Dict[str, callable]:
        """Define derived variable calculations."""
        return {
            'egfr': self._calculate_egfr,
            'ldl_calculated': self._calculate_ldl,
            'mean_arterial_pressure': self._calculate_map,
            'pulse_pressure': self._calculate_pulse_pressure,
            'albumin_creatinine_ratio': self._calculate_acr,
        }
    
    def _calculate_egfr(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate eGFR using CKD-EPI equation."""
        creatinine = data.get('creatinine')
        age = data.get('age')
        sex = data.get('sex')
        race = data.get('race_ethnicity')
        
        if creatinine is None or age is None or sex is None:
            return None
        
        is_female = (sex == 2)
        is_black = (race == 4)
        
        kappa = 0.7 if is_female else 0.9
        alpha = -0.329 if is_female else -0.411
        
        min_ratio = min(creatinine / kappa, 1.0)
        max_ratio = max(creatinine / kappa, 1.0)
        
        egfr = 141 * (min_ratio ** alpha) * (max_ratio ** -1.209) * (0.993 ** age)
        
        if is_female:
            egfr *= 1.018
        if is_black:
            egfr *= 1.159
        
        return egfr
    
    def _calculate_ldl(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate LDL using Friedewald equation."""
        tc = data.get('total_cholesterol')
        hdl = data.get('hdl')
        trig = data.get('triglycerides')
        
        if tc is None or hdl is None or trig is None:
            return None
        
        if trig > 400:
            return None
        
        ldl = tc - hdl - (trig / 5.0)
        return max(0, ldl)
    
    def _calculate_map(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate mean arterial pressure."""
        sbp = data.get('systolic_bp')
        dbp = data.get('diastolic_bp')
        
        if sbp is None or dbp is None:
            return None
        
        return (2 * dbp + sbp) / 3.0
    
    def _calculate_pulse_pressure(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate pulse pressure."""
        sbp = data.get('systolic_bp')
        dbp = data.get('diastolic_bp')
        
        if sbp is None or dbp is None:
            return None
        
        return sbp - dbp
    
    def _calculate_acr(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate albumin-to-creatinine ratio."""
        albumin = data.get('urine_albumin')
        creatinine = data.get('urine_creatinine')
        
        if albumin is None or creatinine is None or creatinine == 0:
            return None
        
        return albumin / creatinine
    
    def map_patient_data(self, demographics: pd.Series, 
                        questionnaire: pd.Series,
                        chemicals: pd.Series) -> Dict[str, Any]:
        """
        Map patient data from NHANES CSV format to standard features.
        
        Args:
            demographics: Patient demographics row
            questionnaire: Patient questionnaire row
            chemicals: Patient lab/chemical row
            
        Returns:
            Dictionary of standardized features
        """
        patient_data = {}
        
        for csv_var, std_var in self.demographics_map.items():
            if csv_var in demographics.index:
                patient_data[std_var] = demographics[csv_var]
        
        for csv_var, std_var in self.questionnaire_map.items():
            if csv_var in questionnaire.index:
                patient_data[std_var] = questionnaire[csv_var]
        
        for csv_var, std_var in self.lab_map.items():
            if csv_var in chemicals.index:
                patient_data[std_var] = chemicals[csv_var]
        
        for derived_var, calc_func in self.derived_calculations.items():
            try:
                patient_data[derived_var] = calc_func(patient_data)
            except Exception:
                patient_data[derived_var] = None
        
        return patient_data
    
    def get_available_variables(self, df: pd.DataFrame) -> List[str]:
        """Get list of available variables in a dataframe."""
        all_vars = set()
        all_vars.update(self.demographics_map.keys())
        all_vars.update(self.lab_map.keys())
        all_vars.update(self.questionnaire_map.keys())
        
        available = [var for var in all_vars if var in df.columns]
        return available
    
    def standardize_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize coded values to meaningful values."""
        standardized = data.copy()
        
        if 'sex' in standardized:
            sex_map = {1: 'male', 2: 'female'}
            standardized['sex'] = sex_map.get(standardized['sex'], standardized['sex'])
        
        if 'race_ethnicity' in standardized:
            race_map = {
                1: 'mexican_american',
                2: 'other_hispanic',
                3: 'non_hispanic_white',
                4: 'non_hispanic_black',
                5: 'other_race'
            }
            standardized['race_ethnicity'] = race_map.get(
                standardized['race_ethnicity'], 
                standardized['race_ethnicity']
            )
        
        yes_no_vars = [
            'ever_smoked', 'current_smoker', 'ever_alcohol',
            'doctor_told_diabetes', 'taking_insulin', 'taking_diabetes_pills',
            'doctor_told_hypertension', 'taking_bp_medication',
            'doctor_told_chf', 'doctor_told_chd', 'doctor_told_angina',
            'doctor_told_heart_attack', 'doctor_told_stroke',
            'doctor_told_cancer', 'doctor_told_kidney_disease',
            'doctor_told_liver_condition'
        ]
        
        for var in yes_no_vars:
            if var in standardized:
                val = standardized[var]
                if val == 1:
                    standardized[var] = True
                elif val == 2:
                    standardized[var] = False
                else:
                    standardized[var] = None
        
        return standardized
    
    def filter_valid_patients(self, df: pd.DataFrame, 
                             required_vars: List[str] = None) -> pd.DataFrame:
        """
        Filter patients with sufficient data for analysis.
        
        Args:
            df: DataFrame with patient data
            required_vars: List of required variables (uses defaults if None)
            
        Returns:
            Filtered DataFrame
        """
        if required_vars is None:
            required_vars = ['age', 'sex', 'SEQN']
        
        mask = pd.Series(True, index=df.index)
        
        for var in required_vars:
            if var in df.columns:
                mask &= df[var].notna()
        
        return df[mask]


def create_variable_search_index(data_dict_path: str) -> pd.DataFrame:
    """
    Create searchable index from NHANES data dictionary.
    
    Args:
        data_dict_path: Path to dictionary CSV file
        
    Returns:
        DataFrame with variable information
    """
    try:
        dict_df = pd.read_csv(data_dict_path)
        
        if 'variable_codename_use' in dict_df.columns:
            dict_df = dict_df.rename(columns={
                'variable_codename_use': 'variable',
                'variable_description_use': 'description',
                'in_dataset': 'dataset'
            })
        
        return dict_df
    except Exception as e:
        print(f"Error loading data dictionary: {e}")
        return pd.DataFrame()


def search_variables(dict_df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """
    Search for variables in data dictionary.
    
    Args:
        dict_df: Data dictionary DataFrame
        search_term: Search term (case-insensitive)
        
    Returns:
        Matching rows
    """
    if dict_df.empty:
        return pd.DataFrame()
    
    mask = pd.Series(False, index=dict_df.index)
    
    search_term = search_term.lower()
    
    for col in ['variable', 'description']:
        if col in dict_df.columns:
            mask |= dict_df[col].astype(str).str.lower().str.contains(search_term, na=False)
    
    return dict_df[mask]


if __name__ == '__main__':
    mapper = NHANESVariableMapper()
    
    print("NHANES Variable Mapper initialized")
    print(f"Demographics variables: {len(mapper.demographics_map)}")
    print(f"Lab variables: {len(mapper.lab_map)}")
    print(f"Questionnaire variables: {len(mapper.questionnaire_map)}")
    print(f"Derived calculations: {len(mapper.derived_calculations)}")
    
    print("\nExample demographics mapping:")
    for csv_var, std_var in list(mapper.demographics_map.items())[:5]:
        print(f"  {csv_var} -> {std_var}")
    
    print("\nExample lab mapping:")
    for csv_var, std_var in list(mapper.lab_map.items())[:10]:
        print(f"  {csv_var} -> {std_var}")
