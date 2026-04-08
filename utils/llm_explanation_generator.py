#!/usr/bin/env python3
"""
LLM-based Explanation Generator

Generates natural language explanations for disease risks and recommendations
using GPT/Claude or local LLMs
"""

import os
from typing import Dict, List, Optional
import json


class LLMExplanationGenerator:
    """
    Generate natural language explanations for digital twin results
    
    Supports multiple LLM backends:
    - OpenAI GPT (API)
    - Anthropic Claude (API)
    - Local models via Ollama
    - Template-based fallback
    """
    
    def __init__(
        self,
        backend: str = "template",
        api_key: Optional[str] = None,
        model: str = "gpt-4"
    ):
        """
        Initialize explanation generator
        
        Args:
            backend: 'openai', 'anthropic', 'ollama', or 'template'
            api_key: API key for OpenAI/Anthropic
            model: Model name
        """
        self.backend = backend
        self.model = model
        
        if backend == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.use_llm = True
                print(f"✓ Using OpenAI {model} for explanations")
            except Exception as e:
                print(f"⚠️ Could not initialize OpenAI: {e}")
                print("   Falling back to template-based explanations")
                self.use_llm = False
        
        elif backend == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
                self.use_llm = True
                print(f"✓ Using Anthropic {model} for explanations")
            except Exception as e:
                print(f"⚠️ Could not initialize Anthropic: {e}")
                print("   Falling back to template-based explanations")
                self.use_llm = False
        
        elif backend == "ollama":
            try:
                import requests
                # Test Ollama connection
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    self.use_llm = True
                    print(f"✓ Using Ollama {model} for explanations")
                else:
                    raise Exception("Ollama not running")
            except Exception as e:
                print(f"⚠️ Could not connect to Ollama: {e}")
                print("   Falling back to template-based explanations")
                self.use_llm = False
        
        else:
            self.use_llm = False
            print("✓ Using template-based explanations")
    
    def generate_patient_report(
        self,
        patient_data: Dict,
        disease_risks: Dict[str, float],
        interventions: List[Dict],
        trajectory_summary: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive patient report
        
        Args:
            patient_data: Patient demographics and lifestyle
            disease_risks: Disease risk predictions
            interventions: Recommended interventions
            trajectory_summary: Optional trajectory analysis
        
        Returns:
            Natural language report
        """
        if self.use_llm and self.backend == "openai":
            return self._generate_with_openai(patient_data, disease_risks, interventions, trajectory_summary)
        elif self.use_llm and self.backend == "anthropic":
            return self._generate_with_anthropic(patient_data, disease_risks, interventions, trajectory_summary)
        elif self.use_llm and self.backend == "ollama":
            return self._generate_with_ollama(patient_data, disease_risks, interventions, trajectory_summary)
        else:
            return self._generate_with_template(patient_data, disease_risks, interventions, trajectory_summary)
    
    def _generate_with_openai(self, patient_data, disease_risks, interventions, trajectory_summary):
        """Generate using OpenAI GPT"""
        prompt = self._build_prompt(patient_data, disease_risks, interventions, trajectory_summary)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant explaining digital twin simulation results to patients. Be clear, empathetic, and actionable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ OpenAI API error: {e}")
            return self._generate_with_template(patient_data, disease_risks, interventions, trajectory_summary)
    
    def _generate_with_anthropic(self, patient_data, disease_risks, interventions, trajectory_summary):
        """Generate using Anthropic Claude"""
        prompt = self._build_prompt(patient_data, disease_risks, interventions, trajectory_summary)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"⚠️ Anthropic API error: {e}")
            return self._generate_with_template(patient_data, disease_risks, interventions, trajectory_summary)
    
    def _generate_with_ollama(self, patient_data, disease_risks, interventions, trajectory_summary):
        """Generate using local Ollama"""
        import requests
        
        prompt = self._build_prompt(patient_data, disease_risks, interventions, trajectory_summary)
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()['response']
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")
            return self._generate_with_template(patient_data, disease_risks, interventions, trajectory_summary)
    
    def _build_prompt(self, patient_data, disease_risks, interventions, trajectory_summary):
        """Build prompt for LLM"""
        # Get top risks
        top_risks = sorted(disease_risks.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = f"""Generate a clear, empathetic medical report for a patient based on their digital twin simulation.

PATIENT PROFILE:
- Age: {patient_data.get('age')} years old
- Sex: {patient_data.get('sex')}
- BMI: {patient_data.get('bmi', 'N/A')}
- Blood Pressure: {patient_data.get('systolic_bp', 'N/A')}/{patient_data.get('diastolic_bp', 'N/A')}
- Glucose: {patient_data.get('fasting_glucose', 'N/A')} mg/dL
- Lifestyle: {'Smoker' if patient_data.get('smoking') else 'Non-smoker'}, Exercise {patient_data.get('exercise_hours_per_week', 0):.1f} hrs/week

10-YEAR DISEASE RISK PREDICTIONS:
"""
        for disease, risk in top_risks:
            if risk > 0.1:
                prompt += f"- {disease.replace('_', ' ').title()}: {risk*100:.1f}% risk\n"
        
        prompt += f"\nTOP RECOMMENDED INTERVENTIONS:\n"
        for i, intervention in enumerate(interventions[:3], 1):
            prompt += f"{i}. {intervention['action']}\n"
            prompt += f"   - Risk reduction: {intervention.get('risk_reduction', 'N/A')}\n"
            prompt += f"   - Timeframe: {intervention.get('timeframe', 'N/A')}\n"
        
        prompt += """
Please generate a report that:
1. Explains the patient's current health status in simple terms
2. Describes which diseases they are at risk for and WHY (based on their lifestyle and biomarkers)
3. Explains how their organs are interacting and affecting each other
4. Provides clear, actionable recommendations with reasoning
5. Is empathetic and encouraging, not alarming

Keep it under 500 words, organized in clear sections.
"""
        return prompt
    
    def _generate_with_template(self, patient_data, disease_risks, interventions, trajectory_summary):
        """Fallback template-based generation"""
        report = []
        report.append("=" * 80)
        report.append("DIGITAL TWIN HEALTH REPORT")
        report.append("=" * 80)
        
        # Patient summary
        report.append("\n📋 YOUR CURRENT HEALTH STATUS\n")
        report.append(f"Based on your profile (age {patient_data.get('age')}, {patient_data.get('sex')}), ")
        report.append(f"our digital twin simulation analyzed your health trajectory over the next 10 years.")
        
        # Identify key risk factors
        risk_factors = []
        if patient_data.get('bmi', 25) > 30:
            risk_factors.append("elevated BMI")
        if patient_data.get('smoking'):
            risk_factors.append("smoking")
        if patient_data.get('systolic_bp', 120) > 140:
            risk_factors.append("high blood pressure")
        if patient_data.get('fasting_glucose', 100) > 100:
            risk_factors.append("elevated glucose")
        if patient_data.get('exercise_hours_per_week', 0) < 2:
            risk_factors.append("insufficient exercise")
        
        if risk_factors:
            report.append(f"\nKey risk factors identified: {', '.join(risk_factors)}.")
        
        # Disease risks
        report.append("\n\n🎯 DISEASE RISK PREDICTIONS (10-YEAR OUTLOOK)\n")
        
        top_risks = sorted(disease_risks.items(), key=lambda x: x[1], reverse=True)
        
        high_risks = [(d, r) for d, r in top_risks if r > 0.5]
        moderate_risks = [(d, r) for d, r in top_risks if 0.3 <= r <= 0.5]
        
        if high_risks:
            report.append("HIGH RISK (>50% probability):")
            for disease, risk in high_risks:
                disease_name = disease.replace('_', ' ').title()
                report.append(f"  • {disease_name}: {risk*100:.1f}% risk")
                
                # Add explanation
                if 'diabetes' in disease.lower():
                    report.append(f"    → Your glucose levels and lifestyle patterns suggest high diabetes risk.")
                elif 'cvd' in disease.lower() or 'cardiovascular' in disease.lower():
                    report.append(f"    → Cardiovascular risk driven by blood pressure, cholesterol, and lifestyle.")
                elif 'hypertension' in disease.lower():
                    report.append(f"    → Blood pressure trends indicate hypertension development.")
        
        if moderate_risks:
            report.append("\nMODERATE RISK (30-50% probability):")
            for disease, risk in moderate_risks[:3]:
                disease_name = disease.replace('_', ' ').title()
                report.append(f"  • {disease_name}: {risk*100:.1f}% risk")
        
        # Organ interactions
        report.append("\n\n🔄 HOW YOUR ORGANS ARE INTERACTING\n")
        report.append("Our simulation shows how your organs influence each other:")
        
        if patient_data.get('fasting_glucose', 100) > 100:
            report.append("  • High glucose levels are affecting your cardiovascular and kidney function")
        if patient_data.get('systolic_bp', 120) > 140:
            report.append("  • Elevated blood pressure is putting strain on your heart and kidneys")
        if patient_data.get('bmi', 25) > 30:
            report.append("  • Excess weight is impacting metabolic, cardiovascular, and liver health")
        if patient_data.get('smoking'):
            report.append("  • Smoking is damaging cardiovascular, respiratory, and immune systems")
        
        # Recommendations
        report.append("\n\n💡 RECOMMENDED ACTIONS\n")
        report.append("Based on your simulation, here's what you can do to reduce your risks:\n")
        
        for i, intervention in enumerate(interventions[:5], 1):
            priority_icon = "🚨" if intervention.get('priority') == 'CRITICAL' else "⚠️" if intervention.get('priority') == 'HIGH' else "ℹ️"
            report.append(f"{i}. {priority_icon} {intervention['action']}")
            
            if 'diseases_affected' in intervention:
                diseases = ', '.join(intervention['diseases_affected'])
                report.append(f"   Reduces risk for: {diseases}")
            
            report.append(f"   Expected impact: {intervention.get('risk_reduction', 'Significant reduction')}")
            report.append(f"   Timeline: {intervention.get('timeframe', 'Varies')}\n")
        
        # Encouragement
        report.append("\n✨ REMEMBER\n")
        report.append("These predictions are based on your current trajectory. The good news is that")
        report.append("lifestyle changes can significantly alter your health outcomes. Small, consistent")
        report.append("changes can make a big difference over time. You have the power to change your")
        report.append("health trajectory!\n")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Singleton instance
_generator_instance = None

def get_explanation_generator(backend: str = "template") -> LLMExplanationGenerator:
    """Get singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LLMExplanationGenerator(backend=backend)
    return _generator_instance


# Example usage
if __name__ == '__main__':
    generator = LLMExplanationGenerator(backend="template")
    
    # Example patient
    patient_data = {
        'age': 45,
        'sex': 'male',
        'bmi': 32.0,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'fasting_glucose': 115,
        'smoking': True,
        'exercise_hours_per_week': 0.0
    }
    
    disease_risks = {
        'diabetes': 0.85,
        'hypertension': 0.90,
        'cvd': 0.75,
        'ckd_stage_3': 0.45,
        'nafld': 0.60
    }
    
    interventions = [
        {
            'action': 'Quit smoking',
            'diseases_affected': ['cvd', 'hypertension'],
            'risk_reduction': '30-40%',
            'timeframe': 'Immediate',
            'priority': 'CRITICAL'
        },
        {
            'action': 'Lose 10% body weight',
            'diseases_affected': ['diabetes', 'hypertension', 'nafld'],
            'risk_reduction': '20-30%',
            'timeframe': '6-12 months',
            'priority': 'HIGH'
        }
    ]
    
    print("Generating patient report...\n")
    report = generator.generate_patient_report(patient_data, disease_risks, interventions)
    print(report)
