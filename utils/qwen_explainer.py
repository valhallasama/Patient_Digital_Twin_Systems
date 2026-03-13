"""
Qwen LLM Integration for Explaining Organ-Level Changes
Uses Alibaba's Qwen-Plus to explain how lifestyle interventions affect each organ
"""

import os
from typing import Dict, List, Any, Optional
import json


class QwenOrganExplainer:
    """
    Use Qwen LLM to explain complex organ interactions and intervention effects
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus"):
        self.api_key = api_key or os.environ.get('QWEN_API_KEY')
        self.model = model
        self.use_llm = self.api_key is not None
        
        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                print("✓ Qwen LLM initialized successfully")
            except ImportError:
                print("⚠️  OpenAI package not found. Install with: pip install openai")
                self.use_llm = False
            except Exception as e:
                print(f"⚠️  Failed to initialize Qwen: {e}")
                self.use_llm = False
        else:
            print("⚠️  No Qwen API key found. Using rule-based explanations.")
            print("   Set QWEN_API_KEY environment variable to enable LLM explanations.")
    
    def explain_organ_changes(
        self,
        organ_name: str,
        baseline_state: Dict[str, Any],
        intervention_state: Dict[str, Any],
        intervention_description: str
    ) -> str:
        """
        Explain how an intervention affected a specific organ
        """
        if self.use_llm:
            return self._llm_explain_organ_changes(
                organ_name, baseline_state, intervention_state, intervention_description
            )
        else:
            return self._rule_based_explain_organ_changes(
                organ_name, baseline_state, intervention_state, intervention_description
            )
    
    def _llm_explain_organ_changes(
        self,
        organ_name: str,
        baseline_state: Dict[str, Any],
        intervention_state: Dict[str, Any],
        intervention_description: str
    ) -> str:
        """
        Use Qwen LLM to generate explanation
        """
        prompt = f"""You are a medical expert explaining how lifestyle interventions affect organ systems.

Organ: {organ_name.upper()}

Intervention Applied: {intervention_description}

Baseline State (before intervention):
{json.dumps(baseline_state, indent=2)}

State After Intervention:
{json.dumps(intervention_state, indent=2)}

Task: Explain in clear, medical terms:
1. What specific changes occurred in this organ
2. WHY these changes happened (physiological mechanisms)
3. How the intervention caused these changes
4. What this means for the patient's health

Keep the explanation concise (3-4 sentences), medically accurate, and focused on the causal chain from intervention → organ changes → health impact.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert specializing in physiology and preventive medicine."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️  Qwen API error: {e}")
            return self._rule_based_explain_organ_changes(
                organ_name, baseline_state, intervention_state, intervention_description
            )
    
    def _rule_based_explain_organ_changes(
        self,
        organ_name: str,
        baseline_state: Dict[str, Any],
        intervention_state: Dict[str, Any],
        intervention_description: str
    ) -> str:
        """
        Fallback rule-based explanation
        """
        # Calculate key changes
        changes = {}
        for key in baseline_state.keys():
            if isinstance(baseline_state[key], (int, float)):
                baseline_val = baseline_state[key]
                intervention_val = intervention_state.get(key, baseline_val)
                if baseline_val != 0:
                    pct_change = ((intervention_val - baseline_val) / baseline_val) * 100
                    if abs(pct_change) > 1:
                        changes[key] = {
                            'baseline': baseline_val,
                            'intervention': intervention_val,
                            'pct_change': pct_change
                        }
        
        # Generate explanation based on organ type
        explanations = {
            'metabolic': self._explain_metabolic_changes,
            'cardiovascular': self._explain_cardiovascular_changes,
            'renal': self._explain_renal_changes,
            'hepatic': self._explain_hepatic_changes,
            'immune': self._explain_immune_changes,
            'endocrine': self._explain_endocrine_changes,
            'neural': self._explain_neural_changes
        }
        
        explain_func = explanations.get(organ_name, lambda c, i: "Changes observed in organ function.")
        return explain_func(changes, intervention_description)
    
    def _explain_metabolic_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, the metabolic system showed: "
        
        if 'glucose' in changes:
            change = changes['glucose']
            if change['pct_change'] < 0:
                explanation += f"Glucose decreased by {abs(change['pct_change']):.1f}% due to improved insulin sensitivity from regular exercise. "
            else:
                explanation += f"Glucose increased by {change['pct_change']:.1f}%, indicating continued metabolic stress. "
        
        if 'insulin_resistance' in changes:
            change = changes['insulin_resistance']
            if change['pct_change'] < 0:
                explanation += f"Insulin resistance improved by {abs(change['pct_change']):.1f}%, as physical activity enhances glucose uptake in muscles. "
        
        if 'hba1c' in changes:
            change = changes['hba1c']
            if change['pct_change'] < 0:
                explanation += f"HbA1c dropped by {abs(change['pct_change']):.1f}%, reflecting better long-term glucose control. "
        
        return explanation.strip()
    
    def _explain_cardiovascular_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, cardiovascular function showed: "
        
        if 'systolic_bp' in changes:
            change = changes['systolic_bp']
            if change['pct_change'] < 0:
                explanation += f"Blood pressure decreased by {abs(change['pct_change']):.1f}% as exercise improves vessel elasticity and reduces vascular resistance. "
        
        if 'atherosclerosis_level' in changes:
            change = changes['atherosclerosis_level']
            if change['pct_change'] < 0:
                explanation += f"Atherosclerosis progression slowed by {abs(change['pct_change']):.1f}% due to improved lipid metabolism and reduced inflammation. "
        
        return explanation.strip()
    
    def _explain_renal_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, kidney function showed: "
        
        if 'egfr' in changes:
            change = changes['egfr']
            if change['pct_change'] > 0:
                explanation += f"eGFR improved by {change['pct_change']:.1f}% as lower blood pressure reduced glomerular stress. "
            else:
                explanation += f"eGFR declined by {abs(change['pct_change']):.1f}%, though at a slower rate than baseline. "
        
        return explanation.strip()
    
    def _explain_hepatic_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, liver function showed: "
        
        if 'ldl' in changes:
            change = changes['ldl']
            if change['pct_change'] < 0:
                explanation += f"LDL cholesterol decreased by {abs(change['pct_change']):.1f}% as improved diet and exercise enhance lipid metabolism. "
        
        if 'alt' in changes:
            change = changes['alt']
            if change['pct_change'] < 0:
                explanation += f"ALT levels improved by {abs(change['pct_change']):.1f}%, indicating reduced liver inflammation. "
        
        return explanation.strip()
    
    def _explain_immune_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, immune function showed: "
        
        if 'inflammation' in changes:
            change = changes['inflammation']
            if change['pct_change'] < 0:
                explanation += f"Systemic inflammation decreased by {abs(change['pct_change']):.1f}% as regular exercise has anti-inflammatory effects. "
        
        return explanation.strip()
    
    def _explain_endocrine_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, endocrine function showed: "
        
        if 'cortisol' in changes:
            change = changes['cortisol']
            if change['pct_change'] < 0:
                explanation += f"Cortisol levels decreased by {abs(change['pct_change']):.1f}% as stress reduction and better sleep normalize HPA axis function. "
        
        return explanation.strip()
    
    def _explain_neural_changes(self, changes: Dict, intervention: str) -> str:
        explanation = f"With {intervention}, neural function showed: "
        
        if 'stress_level' in changes:
            change = changes['stress_level']
            if change['pct_change'] < 0:
                explanation += f"Stress levels decreased by {abs(change['pct_change']):.1f}% through improved sleep quality and regular physical activity. "
        
        return explanation.strip()
    
    def explain_intervention_cascade(
        self,
        intervention_description: str,
        organ_changes: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Explain the cascade of changes across all organs
        """
        if self.use_llm:
            return self._llm_explain_cascade(intervention_description, organ_changes)
        else:
            return self._rule_based_explain_cascade(intervention_description, organ_changes)
    
    def _llm_explain_cascade(
        self,
        intervention_description: str,
        organ_changes: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Use Qwen to explain the cascade of organ interactions
        """
        prompt = f"""You are a medical expert explaining how lifestyle interventions create cascading effects across organ systems.

Intervention: {intervention_description}

Organ Changes Observed:
{json.dumps(organ_changes, indent=2)}

Task: Explain the CASCADE of physiological changes:
1. Which organ responded first and why
2. How changes in one organ affected other organs
3. The chain of cause-and-effect across the body systems
4. Why this cascade ultimately reduced disease risk

Focus on the INTERACTIONS between organs, not just individual changes. Explain it like a story of how the body systems work together. Keep it concise (4-5 sentences).
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert specializing in systems physiology and organ interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️  Qwen API error: {e}")
            return self._rule_based_explain_cascade(intervention_description, organ_changes)
    
    def _rule_based_explain_cascade(
        self,
        intervention_description: str,
        organ_changes: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Fallback rule-based cascade explanation
        """
        explanation = f"The {intervention_description} triggered a cascade of beneficial changes: "
        
        # Typical cascade: Neural → Endocrine → Metabolic → Cardiovascular → Renal
        if 'neural' in organ_changes:
            explanation += "Reduced stress and better sleep normalized the nervous system, "
        
        if 'endocrine' in organ_changes:
            explanation += "which lowered cortisol levels and reduced chronic stress response. "
        
        if 'metabolic' in organ_changes:
            explanation += "This improved insulin sensitivity and glucose metabolism, "
        
        if 'cardiovascular' in organ_changes:
            explanation += "leading to lower blood pressure and reduced vascular stress. "
        
        if 'renal' in organ_changes:
            explanation += "The kidneys benefited from reduced BP and glucose, slowing damage progression. "
        
        explanation += "These interconnected improvements demonstrate how lifestyle changes affect the entire body system, not just individual organs."
        
        return explanation


# Global instance
_explainer = None

def get_qwen_explainer() -> QwenOrganExplainer:
    """Get or create global Qwen explainer"""
    global _explainer
    if _explainer is None:
        _explainer = QwenOrganExplainer()
    return _explainer
