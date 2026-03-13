"""
Medical Report Parser - Extract ALL information from patient reports
Automatically extracts lifestyle, vitals, labs, and history
"""

import re
from typing import Dict, Any, Optional
from datetime import datetime


class MedicalReportParser:
    """
    Comprehensive medical report parser
    Extracts everything: demographics, vitals, labs, lifestyle, history
    """
    
    def __init__(self):
        self.patterns = self._build_extraction_patterns()
    
    def _build_extraction_patterns(self) -> Dict[str, Any]:
        """Build regex patterns for extracting all data"""
        return {
            # Demographics
            'patient_id': r'(?:Patient\s*ID|ID|Patient\s*#):\s*([A-Z0-9-]+)',
            'age': r'(?:Age|age):\s*(\d+)\s*(?:years?|yrs?|y)?',
            'gender': r'(?:Gender|Sex):\s*(Male|Female|M|F)',
            'bmi': r'(?:BMI):\s*(\d+\.?\d*)',
            
            # Vitals
            'bp_systolic': r'(?:BP|Blood\s*Pressure):\s*(\d+)/\d+',
            'bp_diastolic': r'(?:BP|Blood\s*Pressure):\s*\d+/(\d+)',
            'heart_rate': r'(?:Heart\s*Rate|HR|Pulse):\s*(\d+)',
            'temperature': r'(?:Temperature|Temp):\s*(\d+\.?\d*)',
            'respiratory_rate': r'(?:Respiratory\s*Rate|RR):\s*(\d+)',
            
            # Labs - Glucose/Diabetes
            'glucose': r'(?:Fasting\s*)?(?:Glucose|Blood\s*Sugar):\s*(\d+\.?\d*)',
            'hba1c': r'(?:HbA1c|A1C|Hemoglobin\s*A1c):\s*(\d+\.?\d*)',
            'insulin': r'(?:Insulin):\s*(\d+\.?\d*)',
            
            # Labs - Lipids
            'ldl': r'(?:LDL):\s*(\d+\.?\d*)',
            'hdl': r'(?:HDL):\s*(\d+\.?\d*)',
            'triglycerides': r'(?:Triglycerides|TG):\s*(\d+\.?\d*)',
            'total_cholesterol': r'(?:Total\s*Cholesterol|Cholesterol):\s*(\d+\.?\d*)',
            
            # Labs - Kidney
            'creatinine': r'(?:Creatinine):\s*(\d+\.?\d*)',
            'egfr': r'(?:eGFR|GFR):\s*(\d+\.?\d*)',
            'bun': r'(?:BUN):\s*(\d+\.?\d*)',
            
            # Labs - Liver
            'alt': r'(?:ALT):\s*(\d+\.?\d*)',
            'ast': r'(?:AST):\s*(\d+\.?\d*)',
            'bilirubin': r'(?:Bilirubin):\s*(\d+\.?\d*)',
            
            # Labs - Inflammation
            'crp': r'(?:CRP|C-Reactive\s*Protein):\s*(\d+\.?\d*)',
            'wbc': r'(?:WBC|White\s*Blood\s*Cell):\s*(\d+\.?\d*)',
            
            # Lifestyle - Exercise
            'exercise_sessions': r'(?:Exercise|Physical\s*Activity):\s*(\d+)(?:\s*(?:sessions?|times?|days?))?(?:\s*(?:per|/)?\s*week)?',
            'exercise_hours': r'(?:Exercise|Physical\s*Activity):\s*(\d+\.?\d*)\s*(?:hours?|hrs?)',
            'exercise_description': r'(?:Exercise|Physical\s*Activity):\s*([^\n]+)',
            
            # Lifestyle - Sleep
            'sleep_hours': r'(?:Sleep):\s*(\d+\.?\d*)\s*(?:hours?|hrs?|h)',
            'sleep_quality': r'(?:Sleep\s*Quality):\s*(\w+)',
            'sleep_description': r'(?:Sleep):\s*([^\n]+)',
            
            # Lifestyle - Diet
            'diet_description': r'(?:Diet|Nutrition|Eating\s*Habits?):\s*([^\n]+)',
            'calories': r'(?:Calories|Caloric\s*Intake):\s*(\d+)',
            'diet_quality': r'(?:Diet\s*Quality):\s*(\w+)',
            
            # Lifestyle - Stress
            'stress_level': r'(?:Stress\s*Level):\s*(\w+)',
            'stress_description': r'(?:Stress):\s*([^\n]+)',
            
            # Lifestyle - Occupation
            'occupation': r'(?:Occupation|Job|Work):\s*([^\n]+)',
            
            # Lifestyle - Smoking/Alcohol
            'smoking': r'(?:Smoking|Smoker):\s*([^\n]+)',
            'alcohol': r'(?:Alcohol|Drinking):\s*([^\n]+)',
            
            # Medical History
            'family_history': r'(?:Family\s*History):\s*([^\n]+)',
            'medications': r'(?:Medications?|Drugs?):\s*([^\n]+)',
            'allergies': r'(?:Allergies):\s*([^\n]+)',
            'past_conditions': r'(?:Past\s*Medical\s*History|PMH):\s*([^\n]+)',
        }
    
    def parse_report(self, report_text: str) -> Dict[str, Any]:
        """
        Parse complete medical report
        Returns all extracted information
        """
        extracted = {}
        
        # Extract all fields using patterns
        for field, pattern in self.patterns.items():
            match = re.search(pattern, report_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Convert to appropriate type
                if field in ['age', 'heart_rate', 'respiratory_rate', 'exercise_sessions', 'calories']:
                    extracted[field] = int(value)
                elif field in ['bmi', 'bp_systolic', 'bp_diastolic', 'temperature', 'glucose', 
                              'hba1c', 'insulin', 'ldl', 'hdl', 'triglycerides', 'total_cholesterol',
                              'creatinine', 'egfr', 'bun', 'alt', 'ast', 'bilirubin', 'crp', 'wbc',
                              'exercise_hours', 'sleep_hours']:
                    extracted[field] = float(value)
                else:
                    extracted[field] = value
        
        # Infer missing lifestyle data
        extracted = self._infer_lifestyle(extracted)
        
        return extracted
    
    def _infer_lifestyle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer lifestyle parameters from available data"""
        
        # Infer exercise frequency
        if 'exercise_sessions' not in data:
            if 'exercise_description' in data:
                desc = data['exercise_description'].lower()
                if 'sedentary' in desc or 'minimal' in desc or 'rarely' in desc:
                    data['exercise_sessions'] = 1
                elif 'moderate' in desc or 'regular' in desc:
                    data['exercise_sessions'] = 3
                elif 'active' in desc or 'frequent' in desc:
                    data['exercise_sessions'] = 5
                else:
                    # Parse numbers from description
                    numbers = re.findall(r'(\d+)', desc)
                    if numbers:
                        data['exercise_sessions'] = int(numbers[0])
                    else:
                        data['exercise_sessions'] = 1
        
        # Infer sleep hours
        if 'sleep_hours' not in data:
            if 'sleep_description' in data:
                desc = data['sleep_description'].lower()
                numbers = re.findall(r'(\d+\.?\d*)', desc)
                if numbers:
                    data['sleep_hours'] = float(numbers[0])
                elif 'poor' in desc or 'insufficient' in desc:
                    data['sleep_hours'] = 5.5
                elif 'adequate' in desc:
                    data['sleep_hours'] = 7.0
                elif 'good' in desc:
                    data['sleep_hours'] = 8.0
                else:
                    data['sleep_hours'] = 6.5
        
        # Infer diet quality
        if 'diet_quality' not in data:
            if 'diet_description' in data:
                desc = data['diet_description'].lower()
                if any(word in desc for word in ['poor', 'unhealthy', 'fast food', 'processed']):
                    data['diet_quality'] = 'poor'
                elif any(word in desc for word in ['good', 'healthy', 'balanced', 'mediterranean']):
                    data['diet_quality'] = 'good'
                else:
                    data['diet_quality'] = 'moderate'
            else:
                # Infer from BMI
                bmi = data.get('bmi', 25)
                if bmi > 28:
                    data['diet_quality'] = 'poor'
                elif bmi > 25:
                    data['diet_quality'] = 'moderate'
                else:
                    data['diet_quality'] = 'good'
        
        # Infer stress level
        if 'stress_level' not in data:
            if 'stress_description' in data:
                desc = data['stress_description'].lower()
                if any(word in desc for word in ['high', 'severe', 'chronic']):
                    data['stress_level'] = 'high'
                elif any(word in desc for word in ['low', 'minimal']):
                    data['stress_level'] = 'low'
                else:
                    data['stress_level'] = 'moderate'
            else:
                # Infer from occupation
                occupation = data.get('occupation', '').lower()
                if any(word in occupation for word in ['executive', 'manager', 'doctor', 'lawyer']):
                    data['stress_level'] = 'high'
                elif any(word in occupation for word in ['office', 'desk', 'clerk']):
                    data['stress_level'] = 'moderate'
                else:
                    data['stress_level'] = 'low'
        
        # Infer occupation if missing
        if 'occupation' not in data:
            data['occupation'] = 'office_worker'
        
        return data
    
    def extract_lifestyle_profile(self, report_text: str) -> Dict[str, Any]:
        """
        Extract lifestyle profile specifically
        Returns data ready for LifestyleSimulator
        """
        data = self.parse_report(report_text)
        
        # Map to lifestyle profile format
        profile = {
            'occupation': data.get('occupation', 'office_worker'),
            'exercise_sessions_per_week': data.get('exercise_sessions', 1),
            'sleep_hours': data.get('sleep_hours', 6.5),
            'bmi': data.get('bmi', 25),
            'stress_level': data.get('stress_level', 'moderate'),
            'diet_quality': data.get('diet_quality', 'moderate'),
        }
        
        return profile
    
    def get_summary(self, report_text: str) -> str:
        """Get human-readable summary of extracted data"""
        data = self.parse_report(report_text)
        
        summary = "📋 Extracted Patient Information\n"
        summary += "=" * 60 + "\n\n"
        
        # Demographics
        if any(k in data for k in ['patient_id', 'age', 'gender', 'bmi']):
            summary += "👤 Demographics:\n"
            if 'patient_id' in data:
                summary += f"  • Patient ID: {data['patient_id']}\n"
            if 'age' in data:
                summary += f"  • Age: {data['age']} years\n"
            if 'gender' in data:
                summary += f"  • Gender: {data['gender']}\n"
            if 'bmi' in data:
                summary += f"  • BMI: {data['bmi']}\n"
            summary += "\n"
        
        # Vitals
        if any(k in data for k in ['bp_systolic', 'heart_rate']):
            summary += "💓 Vital Signs:\n"
            if 'bp_systolic' in data and 'bp_diastolic' in data:
                summary += f"  • Blood Pressure: {data['bp_systolic']}/{data['bp_diastolic']} mmHg\n"
            if 'heart_rate' in data:
                summary += f"  • Heart Rate: {data['heart_rate']} bpm\n"
            summary += "\n"
        
        # Labs
        if any(k in data for k in ['glucose', 'hba1c', 'ldl', 'creatinine']):
            summary += "🔬 Lab Results:\n"
            if 'glucose' in data:
                summary += f"  • Glucose: {data['glucose']} mmol/L\n"
            if 'hba1c' in data:
                summary += f"  • HbA1c: {data['hba1c']}%\n"
            if 'ldl' in data:
                summary += f"  • LDL: {data['ldl']} mmol/L\n"
            if 'hdl' in data:
                summary += f"  • HDL: {data['hdl']} mmol/L\n"
            if 'creatinine' in data:
                summary += f"  • Creatinine: {data['creatinine']} μmol/L\n"
            if 'alt' in data:
                summary += f"  • ALT: {data['alt']} U/L\n"
            if 'crp' in data:
                summary += f"  • CRP: {data['crp']} mg/L\n"
            summary += "\n"
        
        # Lifestyle
        summary += "🏃 Lifestyle:\n"
        summary += f"  • Exercise: {data.get('exercise_sessions', 1)} sessions/week\n"
        summary += f"  • Sleep: {data.get('sleep_hours', 6.5)} hours/night\n"
        summary += f"  • Diet Quality: {data.get('diet_quality', 'moderate')}\n"
        summary += f"  • Stress Level: {data.get('stress_level', 'moderate')}\n"
        summary += f"  • Occupation: {data.get('occupation', 'office_worker')}\n"
        
        return summary


# Global instance
_parser = None

def get_report_parser() -> MedicalReportParser:
    """Get or create global report parser"""
    global _parser
    if _parser is None:
        _parser = MedicalReportParser()
    return _parser
