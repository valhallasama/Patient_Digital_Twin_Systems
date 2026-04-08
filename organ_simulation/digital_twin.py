#!/usr/bin/env python3
"""
Personalized Digital Twin System

Creates a digital representation of individual patient's organ systems
and simulates their health trajectory with mechanistic explanations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organ_simulation.hybrid_dynamics import HybridOrganDynamics


@dataclass
class DiseaseEvent:
    """Represents a detected disease onset"""
    disease: str
    month: int
    severity: str
    explanation: str
    biomarkers: Dict[str, float]
    contributing_factors: List[str]


@dataclass
class OrganState:
    """Snapshot of organ biomarkers at a timepoint"""
    month: int
    metabolic: Dict[str, float]
    cardiovascular: Dict[str, float]
    liver: Dict[str, float]
    kidney: Dict[str, float]
    immune: Dict[str, float]
    neural: Dict[str, float]
    lifestyle: Dict[str, float]


class DiseaseDetector:
    """
    Detects disease onset based on clinical thresholds
    Generates mechanistic explanations (NOT statistical predictions)
    """
    
    # Clinical thresholds based on medical guidelines
    THRESHOLDS = {
        'fatty_liver': {
            'ALT': 40,  # U/L
            'AST': 40,
            'AST_ALT_ratio': 1.0  # AST/ALT > 1 suggests advanced disease
        },
        'diabetes': {
            'glucose': 126,  # mg/dL fasting
            'HbA1c': 6.5  # %
        },
        'prediabetes': {
            'glucose': 100,
            'HbA1c': 5.7
        },
        'hypertension': {
            'systolic_bp': 140,  # mmHg
            'diastolic_bp': 90
        },
        'kidney_disease': {
            'creatinine': 1.5,  # mg/dL
            'BUN': 25  # mg/dL
        },
        'metabolic_syndrome': {
            'triglycerides': 150,  # mg/dL
            'HDL_low': 40,  # mg/dL (men)
            'glucose': 100,
            'systolic_bp': 130
        }
    }
    
    def check_disease_onset(
        self,
        prev_state: OrganState,
        curr_state: OrganState,
        lifestyle: Dict[str, float]
    ) -> List[DiseaseEvent]:
        """
        Check if any disease thresholds crossed
        Generate mechanistic explanations
        """
        events = []
        
        # Check fatty liver
        fatty_liver_event = self._check_fatty_liver(prev_state, curr_state, lifestyle)
        if fatty_liver_event:
            events.append(fatty_liver_event)
        
        # Check diabetes
        diabetes_event = self._check_diabetes(prev_state, curr_state, lifestyle)
        if diabetes_event:
            events.append(diabetes_event)
        
        # Check hypertension
        hypertension_event = self._check_hypertension(prev_state, curr_state, lifestyle)
        if hypertension_event:
            events.append(hypertension_event)
        
        # Check kidney disease
        kidney_event = self._check_kidney_disease(prev_state, curr_state, lifestyle)
        if kidney_event:
            events.append(kidney_event)
        
        return events
    
    def _check_fatty_liver(
        self,
        prev_state: OrganState,
        curr_state: OrganState,
        lifestyle: Dict[str, float]
    ) -> Optional[DiseaseEvent]:
        """Check for fatty liver onset with mechanistic explanation"""
        
        prev_alt = prev_state.liver.get('ALT', 0)
        curr_alt = curr_state.liver.get('ALT', 0)
        curr_ast = curr_state.liver.get('AST', 0)
        
        threshold = self.THRESHOLDS['fatty_liver']['ALT']
        
        # Disease onset: crossed threshold
        if curr_alt > threshold and prev_alt <= threshold:
            
            # Identify contributing factors (mechanistic)
            factors = []
            
            if lifestyle.get('alcohol_consumption', 0) > 0.7:
                factors.append(f"high alcohol consumption ({lifestyle['alcohol_consumption']:.1f}/1.0)")
            
            if curr_state.metabolic.get('glucose', 0) > 100:
                factors.append(f"elevated glucose ({curr_state.metabolic['glucose']:.1f} mg/dL, metabolic stress)")
            
            if lifestyle.get('exercise_frequency', 0) < 0.3:
                factors.append(f"insufficient exercise ({lifestyle['exercise_frequency']:.1f}/1.0)")
            
            if curr_state.metabolic.get('triglycerides', 0) > 150:
                factors.append(f"high triglycerides ({curr_state.metabolic['triglycerides']:.1f} mg/dL)")
            
            # Calculate rate of change
            alt_increase = curr_alt - prev_alt
            
            # Determine severity
            if curr_alt > 80:
                severity = 'severe'
            elif curr_alt > 60:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            # Generate mechanistic explanation
            explanation = (
                f"Fatty liver detected at month {curr_state.month}. "
                f"ALT elevated to {curr_alt:.1f} U/L (normal <{threshold}). "
                f"Liver enzymes rose {alt_increase:.1f} points due to sustained hepatic stress. "
                f"Contributing factors: {', '.join(factors)}."
            )
            
            if curr_ast / curr_alt > 1.0:
                explanation += f" AST/ALT ratio {curr_ast/curr_alt:.2f} suggests advanced liver damage."
            
            return DiseaseEvent(
                disease='fatty_liver',
                month=curr_state.month,
                severity=severity,
                explanation=explanation,
                biomarkers={'ALT': curr_alt, 'AST': curr_ast},
                contributing_factors=factors
            )
        
        return None
    
    def _check_diabetes(
        self,
        prev_state: OrganState,
        curr_state: OrganState,
        lifestyle: Dict[str, float]
    ) -> Optional[DiseaseEvent]:
        """Check for diabetes onset"""
        
        prev_glucose = prev_state.metabolic.get('glucose', 0)
        curr_glucose = curr_state.metabolic.get('glucose', 0)
        curr_hba1c = curr_state.metabolic.get('HbA1c', 0)
        
        threshold = self.THRESHOLDS['diabetes']['glucose']
        
        if curr_glucose > threshold and prev_glucose <= threshold:
            
            factors = []
            
            if lifestyle.get('diet_quality', 0) < 0.4:
                factors.append(f"poor diet quality ({lifestyle['diet_quality']:.1f}/1.0)")
            
            if lifestyle.get('exercise_frequency', 0) < 0.3:
                factors.append(f"sedentary lifestyle ({lifestyle['exercise_frequency']:.1f}/1.0)")
            
            if curr_state.liver.get('ALT', 0) > 40:
                factors.append(f"liver dysfunction (ALT {curr_state.liver['ALT']:.1f})")
            
            glucose_increase = curr_glucose - prev_glucose
            
            explanation = (
                f"Type 2 diabetes detected at month {curr_state.month}. "
                f"Fasting glucose elevated to {curr_glucose:.1f} mg/dL (diagnostic threshold ≥{threshold}). "
                f"Glucose rose {glucose_increase:.1f} mg/dL due to progressive insulin resistance. "
                f"Contributing factors: {', '.join(factors)}."
            )
            
            if curr_hba1c > 0:
                explanation += f" HbA1c: {curr_hba1c:.1f}%."
            
            return DiseaseEvent(
                disease='diabetes',
                month=curr_state.month,
                severity='moderate' if curr_glucose < 150 else 'severe',
                explanation=explanation,
                biomarkers={'glucose': curr_glucose, 'HbA1c': curr_hba1c},
                contributing_factors=factors
            )
        
        return None
    
    def _check_hypertension(
        self,
        prev_state: OrganState,
        curr_state: OrganState,
        lifestyle: Dict[str, float]
    ) -> Optional[DiseaseEvent]:
        """Check for hypertension onset"""
        
        prev_sbp = prev_state.cardiovascular.get('systolic_bp', 0)
        curr_sbp = curr_state.cardiovascular.get('systolic_bp', 0)
        curr_dbp = curr_state.cardiovascular.get('diastolic_bp', 0)
        
        sbp_threshold = self.THRESHOLDS['hypertension']['systolic_bp']
        
        if curr_sbp > sbp_threshold and prev_sbp <= sbp_threshold:
            
            factors = []
            
            if lifestyle.get('alcohol_consumption', 0) > 0.6:
                factors.append(f"high alcohol intake ({lifestyle['alcohol_consumption']:.1f}/1.0)")
            
            if lifestyle.get('exercise_frequency', 0) < 0.3:
                factors.append(f"insufficient physical activity ({lifestyle['exercise_frequency']:.1f}/1.0)")
            
            if curr_state.kidney.get('creatinine', 0) > 1.2:
                factors.append(f"kidney dysfunction (creatinine {curr_state.kidney['creatinine']:.1f})")
            
            sbp_increase = curr_sbp - prev_sbp
            
            explanation = (
                f"Hypertension detected at month {curr_state.month}. "
                f"Systolic BP elevated to {curr_sbp:.1f} mmHg (threshold ≥{sbp_threshold}). "
                f"Blood pressure rose {sbp_increase:.1f} mmHg due to vascular stress. "
                f"Contributing factors: {', '.join(factors)}."
            )
            
            return DiseaseEvent(
                disease='hypertension',
                month=curr_state.month,
                severity='stage_1' if curr_sbp < 160 else 'stage_2',
                explanation=explanation,
                biomarkers={'systolic_bp': curr_sbp, 'diastolic_bp': curr_dbp},
                contributing_factors=factors
            )
        
        return None
    
    def _check_kidney_disease(
        self,
        prev_state: OrganState,
        curr_state: OrganState,
        lifestyle: Dict[str, float]
    ) -> Optional[DiseaseEvent]:
        """Check for kidney disease onset"""
        
        prev_creat = prev_state.kidney.get('creatinine', 0)
        curr_creat = curr_state.kidney.get('creatinine', 0)
        curr_bun = curr_state.kidney.get('BUN', 0)
        
        threshold = self.THRESHOLDS['kidney_disease']['creatinine']
        
        if curr_creat > threshold and prev_creat <= threshold:
            
            factors = []
            
            if curr_state.cardiovascular.get('systolic_bp', 0) > 140:
                factors.append(f"hypertension (BP {curr_state.cardiovascular['systolic_bp']:.1f})")
            
            if curr_state.metabolic.get('glucose', 0) > 126:
                factors.append(f"diabetes (glucose {curr_state.metabolic['glucose']:.1f})")
            
            creat_increase = curr_creat - prev_creat
            
            explanation = (
                f"Chronic kidney disease detected at month {curr_state.month}. "
                f"Creatinine elevated to {curr_creat:.1f} mg/dL (normal <{threshold}). "
                f"Kidney function declined (creatinine ↑{creat_increase:.1f}) due to sustained renal stress. "
                f"Contributing factors: {', '.join(factors)}."
            )
            
            return DiseaseEvent(
                disease='kidney_disease',
                month=curr_state.month,
                severity='moderate',
                explanation=explanation,
                biomarkers={'creatinine': curr_creat, 'BUN': curr_bun},
                contributing_factors=factors
            )
        
        return None


class DigitalTwin:
    """
    Personalized digital representation of patient's organ systems
    Simulates health trajectory with learned dynamics
    """
    
    def __init__(
        self,
        patient_profile: Dict,
        dynamics_predictor,
        gnn_model,
        transformer_model,
        edge_index,
        device: str = 'cuda'
    ):
        self.patient_id = patient_profile.get('patient_id', 'unknown')
        self.demographics = patient_profile['demographics']
        self.initial_organs = self._normalize_biomarkers(patient_profile['organ_biomarkers'])
        self.baseline_lifestyle = patient_profile['lifestyle']
        self.medications = patient_profile.get('medications', [])
        self.medical_history = patient_profile.get('medical_history', [])
        
        self.dynamics_predictor = dynamics_predictor
        self.gnn_model = gnn_model
        self.transformer_model = transformer_model
        self.edge_index = edge_index
        self.device = device
        
        # Initialize hybrid dynamics system
        self.hybrid_dynamics = HybridOrganDynamics(dynamics_predictor)
        
        self.disease_detector = DiseaseDetector()
        
        # Simulation history
        self.trajectory = []
        self.detected_events = []
        self.monthly_explanations = []  # Store research-based explanations
    
    def _normalize_biomarkers(self, biomarkers: Dict) -> Dict[str, torch.Tensor]:
        """Convert raw lab values to normalized tensors"""
        # TODO: Use same normalization as training data
        # For now, convert to tensors
        normalized = {}
        for organ, values in biomarkers.items():
            if isinstance(values, dict):
                tensor = torch.tensor([v for v in values.values()], dtype=torch.float32)
            else:
                tensor = torch.tensor(values, dtype=torch.float32)
            normalized[organ] = tensor
        return normalized
    
    def simulate_forward(
        self,
        months: int,
        lifestyle_scenario: Dict[str, float],
        intervention_start_month: int = 0,
        verbose: bool = True
    ) -> Tuple[List[OrganState], List[DiseaseEvent]]:
        """
        Simulate patient's health trajectory
        
        Args:
            months: Simulation horizon
            lifestyle_scenario: Lifestyle parameters to apply
            intervention_start_month: When to start intervention
            verbose: Print progress
        
        Returns:
            trajectory: Monthly organ states
            events: Detected disease onsets
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Simulating {months} months for patient {self.patient_id}")
            print(f"{'='*80}\n")
        
        trajectory = []
        events = []
        current_organs = {k: v.clone() for k, v in self.initial_organs.items()}
        
        for month in range(months):
            # Determine lifestyle for this month
            if month >= intervention_start_month:
                lifestyle = lifestyle_scenario
            else:
                lifestyle = self.baseline_lifestyle
            
            # Convert lifestyle to tensor
            lifestyle_tensor = torch.tensor([
                lifestyle.get('exercise_frequency', 0),
                lifestyle.get('alcohol_consumption', 0),
                lifestyle.get('diet_quality', 0),
                lifestyle.get('sleep_hours', 7) / 12,  # Normalize to 0-1
                lifestyle.get('smoking', 0)
            ], dtype=torch.float32).to(self.device)
            
            # Get GNN embeddings (organ interactions)
            with torch.no_grad():
                gnn_emb = self.gnn_model(
                    {k: v.unsqueeze(0).to(self.device) for k, v in current_organs.items()},
                    self.edge_index,
                    return_hidden=True
                )
                gnn_emb_stacked = torch.stack([
                    gnn_emb[organ]
                    for organ in sorted(self.dynamics_predictor.organ_feature_dims.keys())
                ], dim=1)
                
                # Get temporal context
                # TODO: Use actual history
                temporal_emb = torch.zeros(1, self.dynamics_predictor.transformer_dim).to(self.device)
                
                # Use hybrid dynamics system
                # Combines: temporal learning + cross-sectional patterns + domain knowledge
                deltas_dict, explanations = self.hybrid_dynamics.predict_organ_changes(
                    current_organs,
                    lifestyle,
                    self.demographics,
                    gnn_embeddings=gnn_emb_stacked,
                    temporal_context=temporal_emb,
                    time_delta_months=1
                )
                
                # Store explanations for this month
                self.monthly_explanations.append({
                    'month': month,
                    'explanations': explanations
                })
                
                # Convert to format expected by rest of code
                deltas = deltas_dict
            
            # Update organ states
            next_organs = {}
            for organ in current_organs.keys():
                next_organs[organ] = current_organs[organ] + deltas[organ].squeeze(0).cpu()
            
            # Convert to OrganState for disease detection
            curr_state = self._tensors_to_organ_state(next_organs, month, lifestyle)
            
            # Check for disease onset
            if len(trajectory) > 0:
                prev_state = trajectory[-1]
                disease_events = self.disease_detector.check_disease_onset(
                    prev_state, curr_state, lifestyle
                )
                events.extend(disease_events)
                
                if verbose and disease_events:
                    for event in disease_events:
                        print(f"\n⚠️  {event.disease.upper()} DETECTED at month {month}")
                        print(f"   {event.explanation}\n")
            
            trajectory.append(curr_state)
            current_organs = next_organs
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Simulation complete: {len(events)} disease events detected")
            print(f"{'='*80}\n")
        
        return trajectory, events
    
    def _tensors_to_organ_state(
        self,
        organ_tensors: Dict[str, torch.Tensor],
        month: int,
        lifestyle: Dict[str, float]
    ) -> OrganState:
        """Convert tensor representation to OrganState"""
        # TODO: Properly map tensor indices to biomarker names
        return OrganState(
            month=month,
            metabolic={'glucose': organ_tensors['metabolic'][0].item(), 
                      'HbA1c': organ_tensors['metabolic'][1].item() if len(organ_tensors['metabolic']) > 1 else 0,
                      'triglycerides': organ_tensors['metabolic'][2].item() if len(organ_tensors['metabolic']) > 2 else 0},
            cardiovascular={'systolic_bp': organ_tensors['cardiovascular'][0].item(),
                          'diastolic_bp': organ_tensors['cardiovascular'][1].item() if len(organ_tensors['cardiovascular']) > 1 else 0},
            liver={'ALT': organ_tensors['liver'][0].item(),
                  'AST': organ_tensors['liver'][1].item() if len(organ_tensors['liver']) > 1 else 0},
            kidney={'creatinine': organ_tensors['kidney'][0].item(),
                   'BUN': organ_tensors['kidney'][1].item() if len(organ_tensors['kidney']) > 1 else 0},
            immune={'WBC': organ_tensors['immune'][0].item()},
            neural={'cognitive_score': organ_tensors['neural'][0].item()},
            lifestyle=lifestyle
        )


class InterventionAnalyzer:
    """Compare different lifestyle/treatment scenarios"""
    
    def analyze_scenarios(
        self,
        digital_twin: DigitalTwin,
        scenarios: Dict[str, Dict[str, float]],
        months: int = 24
    ) -> str:
        """
        Simulate multiple futures and generate comparison report
        
        Args:
            digital_twin: Patient's digital twin
            scenarios: Dict of {scenario_name: lifestyle_params}
            months: Simulation horizon
        
        Returns:
            Formatted comparison report
        """
        print(f"\n{'='*80}")
        print(f"INTERVENTION SCENARIO ANALYSIS")
        print(f"Patient: {digital_twin.patient_id}")
        print(f"Simulation horizon: {months} months")
        print(f"{'='*80}\n")
        
        results = {}
        
        for name, lifestyle in scenarios.items():
            print(f"\nSimulating scenario: {name}...")
            trajectory, events = digital_twin.simulate_forward(
                months=months,
                lifestyle_scenario=lifestyle,
                verbose=False
            )
            
            results[name] = {
                'trajectory': trajectory,
                'events': events,
                'disease_free_months': self._count_disease_free_months(events, months)
            }
        
        return self._generate_report(results, months)
    
    def _count_disease_free_months(self, events: List[DiseaseEvent], total_months: int) -> int:
        """Count months without disease"""
        if not events:
            return total_months
        first_event_month = min(e.month for e in events)
        return first_event_month
    
    def _generate_report(self, results: Dict, months: int) -> str:
        """Generate human-readable comparison report"""
        report = []
        
        report.append(f"\n{'='*80}")
        report.append("SCENARIO COMPARISON REPORT")
        report.append(f"{'='*80}\n")
        
        for scenario, data in results.items():
            report.append(f"\n## Scenario: {scenario}")
            report.append(f"   Disease-free months: {data['disease_free_months']}/{months}")
            
            if data['events']:
                report.append(f"\n   Disease events detected:")
                for event in data['events']:
                    report.append(f"\n   📍 Month {event.month}: {event.disease.upper()} ({event.severity})")
                    report.append(f"      {event.explanation}")
            else:
                report.append(f"\n   ✅ No disease events detected")
            
            # Show key biomarker trends for all organs
            initial = data['trajectory'][0]
            final = data['trajectory'][-1]
            
            report.append(f"\n   Biomarker changes over {months} months:")
            
            # Liver
            report.append(f"\n   Liver:")
            report.append(f"      ALT: {initial.liver.get('ALT', 0):.1f} → {final.liver.get('ALT', 0):.1f} U/L (change: {final.liver.get('ALT', 0) - initial.liver.get('ALT', 0):+.1f})")
            report.append(f"      AST: {initial.liver.get('AST', 0):.1f} → {final.liver.get('AST', 0):.1f} U/L (change: {final.liver.get('AST', 0) - initial.liver.get('AST', 0):+.1f})")
            
            # Metabolic
            report.append(f"\n   Metabolic:")
            report.append(f"      Glucose: {initial.metabolic.get('glucose', 0):.1f} → {final.metabolic.get('glucose', 0):.1f} mg/dL (change: {final.metabolic.get('glucose', 0) - initial.metabolic.get('glucose', 0):+.1f})")
            report.append(f"      HbA1c: {initial.metabolic.get('HbA1c', 0):.2f} → {final.metabolic.get('HbA1c', 0):.2f} % (change: {final.metabolic.get('HbA1c', 0) - initial.metabolic.get('HbA1c', 0):+.2f})")
            report.append(f"      Triglycerides: {initial.metabolic.get('triglycerides', 0):.1f} → {final.metabolic.get('triglycerides', 0):.1f} mg/dL (change: {final.metabolic.get('triglycerides', 0) - initial.metabolic.get('triglycerides', 0):+.1f})")
            
            # Cardiovascular
            report.append(f"\n   Cardiovascular:")
            report.append(f"      Systolic BP: {initial.cardiovascular.get('systolic_bp', 0):.1f} → {final.cardiovascular.get('systolic_bp', 0):.1f} mmHg (change: {final.cardiovascular.get('systolic_bp', 0) - initial.cardiovascular.get('systolic_bp', 0):+.1f})")
            report.append(f"      Diastolic BP: {initial.cardiovascular.get('diastolic_bp', 0):.1f} → {final.cardiovascular.get('diastolic_bp', 0):.1f} mmHg (change: {final.cardiovascular.get('diastolic_bp', 0) - initial.cardiovascular.get('diastolic_bp', 0):+.1f})")
            
            # Kidney
            report.append(f"\n   Kidney:")
            report.append(f"      Creatinine: {initial.kidney.get('creatinine', 0):.2f} → {final.kidney.get('creatinine', 0):.2f} mg/dL (change: {final.kidney.get('creatinine', 0) - initial.kidney.get('creatinine', 0):+.2f})")
            report.append("")
        
        report.append(f"{'='*80}\n")
        
        return "\n".join(report)
