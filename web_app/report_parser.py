#!/usr/bin/env python3
"""
Medical Report Parser
Extracts structured data from free-text medical reports using NLP
"""

import re
from typing import Dict, Optional, List
import json


class MedicalReportParser:
    """Parse medical reports to extract patient data"""
    
    def __init__(self):
        # Regex patterns for common medical values
        self.patterns = {
            'age': [
                r'age[:\s]+(\d+)',
                r'(\d+)\s*(?:year|yr|y\.o\.|years old)',
                r'(\d+)\s*y/o'
            ],
            'sex': [
                r'sex[:\s]+(male|female|m|f)',
                r'gender[:\s]+(male|female|m|f)',
                r'\b(male|female)\b'
            ],
            'height': [
                r'height[:\s]+(\d+\.?\d*)\s*cm',
                r'(\d+\.?\d*)\s*cm\s*tall',
                r'ht[:\s]+(\d+\.?\d*)\s*cm'
            ],
            'weight': [
                r'weight[:\s]+(\d+\.?\d*)\s*kg',
                r'(\d+\.?\d*)\s*kg\b',
                r'wt[:\s]+(\d+\.?\d*)\s*kg'
            ],
            'bmi': [
                r'bmi[:\s]+(\d+\.?\d*)',
                r'body mass index[:\s]+(\d+\.?\d*)'
            ],
            'hba1c': [
                r'hba1c[:\s]+(\d+\.?\d*)%?',
                r'hemoglobin a1c[:\s]+(\d+\.?\d*)%?',
                r'glycated hemoglobin[:\s]+(\d+\.?\d*)%?'
            ],
            'glucose': [
                r'fasting glucose[:\s]+(\d+\.?\d*)',
                r'blood glucose[:\s]+(\d+\.?\d*)',
                r'glucose[:\s]+(\d+\.?\d*)\s*mg/dl'
            ],
            'systolic_bp': [
                r'blood pressure[:\s]+(\d+)/\d+',
                r'bp[:\s]+(\d+)/\d+',
                r'(\d+)/\d+\s*mmhg'
            ],
            'diastolic_bp': [
                r'blood pressure[:\s]+\d+/(\d+)',
                r'bp[:\s]+\d+/(\d+)',
                r'\d+/(\d+)\s*mmhg'
            ],
            'total_cholesterol': [
                r'total cholesterol[:\s]+(\d+\.?\d*)',
                r'cholesterol[:\s]+(\d+\.?\d*)\s*mg/dl'
            ],
            'ldl': [
                r'ldl[:\s]+(\d+\.?\d*)',
                r'ldl cholesterol[:\s]+(\d+\.?\d*)',
                r'low-density lipoprotein[:\s]+(\d+\.?\d*)'
            ],
            'hdl': [
                r'hdl[:\s]+(\d+\.?\d*)',
                r'hdl cholesterol[:\s]+(\d+\.?\d*)',
                r'high-density lipoprotein[:\s]+(\d+\.?\d*)'
            ],
            'triglycerides': [
                r'triglycerides[:\s]+(\d+\.?\d*)',
                r'tg[:\s]+(\d+\.?\d*)'
            ],
            'creatinine': [
                r'creatinine[:\s]+(\d+\.?\d*)',
                r'cr[:\s]+(\d+\.?\d*)\s*mg/dl'
            ],
            'alt': [
                r'alt[:\s]+(\d+\.?\d*)',
                r'alanine aminotransferase[:\s]+(\d+\.?\d*)'
            ],
            'ast': [
                r'ast[:\s]+(\d+\.?\d*)',
                r'aspartate aminotransferase[:\s]+(\d+\.?\d*)'
            ],
            'crp': [
                r'crp[:\s]+(\d+\.?\d*)',
                r'c-reactive protein[:\s]+(\d+\.?\d*)'
            ]
        }
        
        # Lifestyle patterns
        self.lifestyle_patterns = {
            'smoking': {
                'never': r'non-?smoker|never smoked|no smoking',
                'former': r'former smoker|quit smoking|ex-smoker',
                'current': r'current smoker|smokes|smoking'
            },
            'physical_activity': {
                'sedentary': r'sedentary|no exercise|inactive',
                'light': r'light activity|walks occasionally',
                'moderate': r'moderate exercise|regular walking',
                'vigorous': r'vigorous exercise|athletic|runs regularly'
            },
            'diet_quality': {
                'poor': r'poor diet|unhealthy eating',
                'fair': r'fair diet|average diet',
                'good': r'good diet|healthy eating',
                'excellent': r'excellent diet|very healthy'
            }
        }
    
    def parse_report(self, report_text: str) -> Dict:
        """
        Parse a medical report and extract structured data
        
        Args:
            report_text: Free-text medical report
            
        Returns:
            Dictionary with extracted patient data
        """
        # Convert to lowercase for matching
        text_lower = report_text.lower()
        
        patient_data = {
            'patient_id': 'REPORT_UPLOAD',
            'source': 'parsed_report'
        }
        
        # Extract numeric values
        for field, patterns in self.patterns.items():
            value = self._extract_value(text_lower, patterns)
            if value is not None:
                # Handle special cases
                if field == 'sex':
                    patient_data['sex'] = 'M' if value.lower() in ['male', 'm'] else 'F'
                elif field in ['systolic_bp', 'diastolic_bp']:
                    if 'blood_pressure' not in patient_data:
                        patient_data['blood_pressure'] = {}
                    key = 'systolic' if field == 'systolic_bp' else 'diastolic'
                    patient_data['blood_pressure'][key] = float(value)
                elif field in ['ldl', 'hdl', 'total_cholesterol', 'triglycerides']:
                    patient_data[f'{field}_cholesterol' if field in ['ldl', 'hdl'] else field] = float(value)
                else:
                    try:
                        patient_data[field] = float(value) if '.' in str(value) else int(value)
                    except:
                        patient_data[field] = value
        
        # Extract lifestyle factors
        lifestyle = {}
        for category, patterns in self.lifestyle_patterns.items():
            for value, pattern in patterns.items():
                if re.search(pattern, text_lower):
                    lifestyle[category] = value
                    break
        
        if lifestyle:
            patient_data['lifestyle'] = lifestyle
        
        # Extract family history
        family_history = {}
        if re.search(r'family history.*diabetes', text_lower):
            family_history['diabetes'] = True
        if re.search(r'family history.*(heart disease|cardiovascular)', text_lower):
            family_history['cardiovascular_disease'] = True
        
        if family_history:
            patient_data['family_history'] = family_history
        
        return patient_data
    
    def _extract_value(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract first matching value from text using patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def get_completeness_score(self, patient_data: Dict) -> float:
        """Calculate how complete the extracted data is"""
        essential_fields = ['age', 'sex', 'height', 'weight', 'hba1c', 
                           'glucose', 'blood_pressure', 'ldl_cholesterol']
        
        found = sum(1 for field in essential_fields if field in patient_data)
        return found / len(essential_fields)
    
    def format_extraction_summary(self, patient_data: Dict) -> str:
        """Create a human-readable summary of extracted data"""
        summary = "📋 **Extracted Data:**\n\n"
        
        if 'age' in patient_data:
            summary += f"- Age: {patient_data['age']} years\n"
        if 'sex' in patient_data:
            summary += f"- Sex: {patient_data['sex']}\n"
        if 'height' in patient_data and 'weight' in patient_data:
            summary += f"- Height/Weight: {patient_data['height']} cm / {patient_data['weight']} kg\n"
        if 'hba1c' in patient_data:
            summary += f"- HbA1c: {patient_data['hba1c']}%\n"
        if 'glucose' in patient_data:
            summary += f"- Glucose: {patient_data['glucose']} mg/dL\n"
        if 'blood_pressure' in patient_data:
            bp = patient_data['blood_pressure']
            summary += f"- Blood Pressure: {bp.get('systolic', '?')}/{bp.get('diastolic', '?')} mmHg\n"
        if 'ldl_cholesterol' in patient_data:
            summary += f"- LDL: {patient_data['ldl_cholesterol']} mg/dL\n"
        if 'hdl_cholesterol' in patient_data:
            summary += f"- HDL: {patient_data['hdl_cholesterol']} mg/dL\n"
        
        completeness = self.get_completeness_score(patient_data)
        summary += f"\n**Data Completeness:** {completeness*100:.0f}%\n"
        
        if completeness < 0.5:
            summary += "\n⚠️ Limited data extracted. Missing values will be imputed."
        
        return summary


# Example usage
if __name__ == "__main__":
    parser = MedicalReportParser()
    
    # Test with sample report
    sample_report = """
    Patient: John Doe
    Age: 45 years old
    Sex: Male
    Height: 175 cm
    Weight: 85 kg
    
    Lab Results:
    - HbA1c: 6.2%
    - Fasting Glucose: 115 mg/dL
    - Blood Pressure: 140/90 mmHg
    - Total Cholesterol: 220 mg/dL
    - LDL: 145 mg/dL
    - HDL: 42 mg/dL
    - Triglycerides: 180 mg/dL
    
    Lifestyle:
    - Current smoker
    - Sedentary lifestyle
    - Family history of diabetes
    """
    
    data = parser.parse_report(sample_report)
    print(json.dumps(data, indent=2))
    print("\n" + parser.format_extraction_summary(data))
