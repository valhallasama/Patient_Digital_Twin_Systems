import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    def __init__(self):
        self.unit_conversions = {
            'glucose': {
                'mg_dl_to_mmol_l': lambda x: x / 18.0,
                'mmol_l_to_mg_dl': lambda x: x * 18.0
            },
            'cholesterol': {
                'mg_dl_to_mmol_l': lambda x: x / 38.67,
                'mmol_l_to_mg_dl': lambda x: x * 38.67
            },
            'weight': {
                'lb_to_kg': lambda x: x * 0.453592,
                'kg_to_lb': lambda x: x / 0.453592
            },
            'height': {
                'inch_to_cm': lambda x: x * 2.54,
                'cm_to_inch': lambda x: x / 2.54
            },
            'temperature': {
                'f_to_c': lambda x: (x - 32) * 5/9,
                'c_to_f': lambda x: x * 9/5 + 32
            }
        }
        
        self.standard_ranges = {
            'glucose_mmol_l': (3.5, 15.0),
            'hba1c_percent': (4.0, 14.0),
            'total_cholesterol_mmol_l': (2.0, 12.0),
            'ldl_cholesterol_mmol_l': (1.0, 8.0),
            'hdl_cholesterol_mmol_l': (0.5, 3.0),
            'triglycerides_mmol_l': (0.3, 6.0),
            'systolic_bp': (80, 220),
            'diastolic_bp': (50, 140),
            'heart_rate': (40, 150),
            'bmi': (15, 50),
            'age': (0, 120),
            'weight_kg': (30, 200),
            'height_cm': (100, 220)
        }
    
    def normalize_glucose(self, value: float, from_unit: str = 'mg_dl') -> float:
        if from_unit == 'mg_dl':
            return self.unit_conversions['glucose']['mg_dl_to_mmol_l'](value)
        return value
    
    def normalize_cholesterol(self, value: float, from_unit: str = 'mg_dl') -> float:
        if from_unit == 'mg_dl':
            return self.unit_conversions['cholesterol']['mg_dl_to_mmol_l'](value)
        return value
    
    def normalize_weight(self, value: float, from_unit: str = 'lb') -> float:
        if from_unit == 'lb':
            return self.unit_conversions['weight']['lb_to_kg'](value)
        return value
    
    def normalize_height(self, value: float, from_unit: str = 'inch') -> float:
        if from_unit == 'inch':
            return self.unit_conversions['height']['inch_to_cm'](value)
        return value
    
    def normalize_temperature(self, value: float, from_unit: str = 'f') -> float:
        if from_unit == 'f':
            return self.unit_conversions['temperature']['f_to_c'](value)
        return value
    
    def calculate_bmi(self, weight_kg: float, height_cm: float) -> float:
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    
    def clip_to_range(self, value: float, field_name: str) -> float:
        if field_name in self.standard_ranges:
            min_val, max_val = self.standard_ranges[field_name]
            return np.clip(value, min_val, max_val)
        return value
    
    def normalize_dataframe(self, df: pd.DataFrame, 
                           field_mappings: Optional[Dict] = None) -> pd.DataFrame:
        df_normalized = df.copy()
        
        if field_mappings is None:
            field_mappings = {}
        
        for field, config in field_mappings.items():
            if field in df_normalized.columns:
                from_unit = config.get('from_unit')
                target_field = config.get('target_field', field)
                
                if 'glucose' in field.lower():
                    df_normalized[target_field] = df_normalized[field].apply(
                        lambda x: self.normalize_glucose(x, from_unit) if pd.notna(x) else x
                    )
                elif 'cholesterol' in field.lower():
                    df_normalized[target_field] = df_normalized[field].apply(
                        lambda x: self.normalize_cholesterol(x, from_unit) if pd.notna(x) else x
                    )
                elif 'weight' in field.lower():
                    df_normalized[target_field] = df_normalized[field].apply(
                        lambda x: self.normalize_weight(x, from_unit) if pd.notna(x) else x
                    )
                elif 'height' in field.lower():
                    df_normalized[target_field] = df_normalized[field].apply(
                        lambda x: self.normalize_height(x, from_unit) if pd.notna(x) else x
                    )
        
        if 'weight_kg' in df_normalized.columns and 'height_cm' in df_normalized.columns:
            df_normalized['bmi'] = df_normalized.apply(
                lambda row: self.calculate_bmi(row['weight_kg'], row['height_cm'])
                if pd.notna(row['weight_kg']) and pd.notna(row['height_cm']) else np.nan,
                axis=1
            )
        
        for field in self.standard_ranges.keys():
            if field in df_normalized.columns:
                df_normalized[field] = df_normalized[field].apply(
                    lambda x: self.clip_to_range(x, field) if pd.notna(x) else x
                )
        
        return df_normalized
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            'blood_glucose': 'glucose_mmol_l',
            'glucose': 'glucose_mmol_l',
            'a1c': 'hba1c_percent',
            'hemoglobin_a1c': 'hba1c_percent',
            'cholesterol': 'total_cholesterol_mmol_l',
            'bp_systolic': 'systolic_bp',
            'bp_diastolic': 'diastolic_bp',
            'pulse': 'heart_rate',
            'hr': 'heart_rate',
            'body_mass_index': 'bmi',
            'sex': 'gender',
            'smoking': 'smoking_status'
        }
        
        df_renamed = df.copy()
        
        for old_name, new_name in column_mapping.items():
            if old_name in df_renamed.columns:
                df_renamed.rename(columns={old_name: new_name}, inplace=True)
        
        df_renamed.columns = df_renamed.columns.str.lower().str.replace(' ', '_')
        
        return df_renamed


if __name__ == "__main__":
    normalizer = DataNormalizer()
    
    test_data = pd.DataFrame({
        'blood_glucose': [180, 200, 150],
        'cholesterol': [250, 280, 220],
        'weight': [180, 200, 160],
        'height': [70, 68, 72],
        'bp_systolic': [140, 150, 130],
        'sex': ['M', 'F', 'M']
    })
    
    logger.info("Original data:")
    logger.info(test_data)
    
    field_mappings = {
        'blood_glucose': {'from_unit': 'mg_dl', 'target_field': 'glucose_mmol_l'},
        'cholesterol': {'from_unit': 'mg_dl', 'target_field': 'total_cholesterol_mmol_l'},
        'weight': {'from_unit': 'lb', 'target_field': 'weight_kg'},
        'height': {'from_unit': 'inch', 'target_field': 'height_cm'}
    }
    
    normalized = normalizer.normalize_dataframe(test_data, field_mappings)
    normalized = normalizer.standardize_column_names(normalized)
    
    logger.info("\nNormalized data:")
    logger.info(normalized)
