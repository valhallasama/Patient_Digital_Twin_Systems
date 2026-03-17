#!/usr/bin/env python3
"""
LLM Service Integration for Web App
Connects LLM interpreter to Flask backend
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_integration.llm_interpreter import LLMInterpreter
from typing import Dict, List


class LLMService:
    """Service layer for LLM integration in web app"""
    
    def __init__(self):
        """Initialize LLM service"""
        self.interpreter = LLMInterpreter(provider="openai", model="gpt-4")
    
    def analyze_patient(self, patient_data: Dict) -> Dict:
        """Analyze patient data with LLM"""
        return self.interpreter.interpret_patient_data(patient_data)
    
    def explain_results(self, patient_data: Dict, trajectory: List[Dict], predictions: List[Dict]) -> Dict:
        """Explain simulation results with LLM"""
        return self.interpreter.explain_simulation_results(patient_data, trajectory, predictions)
    
    def get_recommendations(self, patient_data: Dict, predictions: List[Dict]) -> Dict:
        """Get personalized recommendations"""
        return self.interpreter.generate_recommendations(patient_data, predictions)
    
    def get_guidelines(self, patient_data: Dict, predictions: List[Dict]) -> Dict:
        """Get clinical guidelines"""
        return self.interpreter.get_clinical_guidelines(patient_data, predictions)
    
    def generate_patient_report(self, patient_data: Dict, predictions: List[Dict], recommendations: Dict) -> str:
        """Generate patient-friendly report"""
        return self.interpreter.generate_patient_report(patient_data, predictions, recommendations)
    
    def explain_risk(self, disease: str, prediction: Dict, patient_data: Dict) -> str:
        """Explain specific risk"""
        return self.interpreter.explain_specific_risk(disease, prediction, patient_data)


# Global service instance
llm_service = LLMService()
