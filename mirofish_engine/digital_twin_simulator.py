#!/usr/bin/env python3
"""
Multi-Year Health Trajectory Simulator
MiroFish-style simulation engine for medical digital twin
"""

import numpy as np
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from datetime import datetime, timedelta

from .comprehensive_agents import (
    MetabolicAgent,
    CardiovascularAgent,
    HepaticAgent,
    RenalAgent,
    ImmuneAgent,
    NeuralAgent,
    EndocrineAgent
)


class DigitalTwinSimulator:
    """
    Simulates health trajectory over multiple years
    MiroFish-style multi-agent system
    """
    
    def __init__(self, patient_data: Dict):
        """
        Initialize digital twin from patient data
        Handles missing data via medical theory
        """
        self.patient_data = patient_data
        self.patient_id = patient_data.get('patient_id', 'unknown')
        
        # Create all 7 agents
        self.agents = {
            'metabolic': MetabolicAgent(patient_data),
            'cardiovascular': CardiovascularAgent(patient_data),
            'hepatic': HepaticAgent(patient_data),
            'renal': RenalAgent(patient_data),
            'immune': ImmuneAgent(patient_data),
            'neural': NeuralAgent(patient_data),
            'endocrine': EndocrineAgent(patient_data)
        }
        
        # Simulation state
        self.timeline = []
        self.disease_predictions = []
        self.current_time = 0
        self.start_date = datetime.now()
        
        # Environment (lifestyle, stress, etc.)
        self.environment = self._initialize_environment(patient_data)
    
    def _initialize_environment(self, data: Dict) -> Dict:
        """Initialize environmental factors"""
        lifestyle = data.get('lifestyle', {})
        
        return {
            'exercise_level': self._lifestyle_to_numeric(
                lifestyle.get('physical_activity', 'moderate')
            ),
            'diet_quality': self._lifestyle_to_numeric(
                lifestyle.get('diet_quality', 'fair')
            ),
            'stress_level': self._lifestyle_to_numeric(
                lifestyle.get('stress_level', 'moderate')
            ),
            'sleep_quality': lifestyle.get('sleep_duration', 7) / 8,  # 0-1 scale
            'smoking': 1.0 if lifestyle.get('smoking_status') == 'current' else 0.0,
            'alcohol': self._lifestyle_to_numeric(
                lifestyle.get('alcohol_consumption', 'none')
            )
        }
    
    def _lifestyle_to_numeric(self, value: str) -> float:
        """Convert lifestyle categories to numeric 0-1"""
        mapping = {
            # Activity levels
            'sedentary': 0.0,
            'light': 0.3,
            'moderate': 0.6,
            'vigorous': 1.0,
            
            # Quality levels
            'poor': 0.2,
            'fair': 0.5,
            'good': 0.8,
            'excellent': 1.0,
            
            # Stress levels
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.9,
            
            # Alcohol
            'none': 0.0,
            'light': 0.3,
            'moderate': 0.6,
            'heavy': 1.0
        }
        
        return mapping.get(value.lower(), 0.5)
    
    def simulate(self, years: int = 5, timestep: str = 'month') -> Dict:
        """
        Simulate health trajectory
        
        Args:
            years: Number of years to simulate
            timestep: 'day', 'week', or 'month'
        
        Returns:
            Complete simulation results
        """
        # Calculate number of steps
        steps_per_year = {
            'day': 365,
            'week': 52,
            'month': 12
        }
        
        total_steps = years * steps_per_year[timestep]
        
        print(f"\n🔬 Starting {years}-year simulation ({total_steps} {timestep}s)...")
        print(f"   Patient ID: {self.patient_id}")
        print(f"   Agents: {len(self.agents)}")
        
        # Simulation loop
        for step in range(total_steps):
            # Update environment (seasonal variations, lifestyle changes)
            self._update_environment(step, timestep)
            
            # Gather signals from all agents
            signals = self._gather_signals()
            
            # Each agent perceives environment and other agents
            for agent_name, agent in self.agents.items():
                agent.perceive({
                    **signals,
                    **self.environment
                })
            
            # Each agent acts (updates internal state)
            for agent_name, agent in self.agents.items():
                agent.act()
            
            # Check for disease emergence
            if step % steps_per_year[timestep] == 0:  # Check yearly
                self._check_disease_emergence(step / steps_per_year[timestep])
            
            # Record state
            if step % (steps_per_year[timestep] // 4) == 0:  # Record quarterly
                self._record_state(step, timestep)
            
            # Progress indicator
            if (step + 1) % max(1, total_steps // 10) == 0:
                progress = (step + 1) / total_steps * 100
                print(f"   Progress: {progress:.0f}% ({step + 1}/{total_steps} {timestep}s)")
        
        print("   ✓ Simulation complete!")
        
        # Generate final predictions
        self._generate_final_predictions()
        
        # Return complete results
        return self._compile_results()
    
    def _update_environment(self, step: int, timestep: str):
        """Update environmental factors over time"""
        # Add some natural variation
        # Stress varies seasonally
        season_factor = np.sin(step / 12 * 2 * np.pi) * 0.1
        self.environment['stress_level'] = max(0, min(1, 
            self.environment['stress_level'] + season_factor
        ))
        
        # Exercise might decrease slightly with age
        self.environment['exercise_level'] *= 0.9999
    
    def _gather_signals(self) -> Dict:
        """Gather signals from all agents"""
        signals = {}
        
        for agent_name, agent in self.agents.items():
            agent_signals = agent.act()
            signals.update(agent_signals)
        
        return signals
    
    def _check_disease_emergence(self, year: float):
        """Check if any diseases are emerging"""
        year_predictions = []
        
        for agent_name, agent in self.agents.items():
            prediction = agent.predict_disease()
            
            # Handle both single predictions and lists
            if isinstance(prediction, list):
                for pred in prediction:
                    if pred['probability'] > 0.3:  # Threshold for reporting
                        year_predictions.append({
                            'year': year,
                            'agent': agent_name,
                            **pred
                        })
            else:
                if prediction['probability'] > 0.3:
                    year_predictions.append({
                        'year': year,
                        'agent': agent_name,
                        **prediction
                    })
        
        if year_predictions:
            self.disease_predictions.extend(year_predictions)
    
    def _record_state(self, step: int, timestep: str):
        """Record current state of all agents"""
        state_snapshot = {
            'step': step,
            'timestep': timestep,
            'date': self._calculate_date(step, timestep),
            'agents': {}
        }
        
        for agent_name, agent in self.agents.items():
            # Get last recorded state from agent
            if agent.state.history:
                state_snapshot['agents'][agent_name] = agent.state.history[-1]
        
        self.timeline.append(state_snapshot)
    
    def _calculate_date(self, step: int, timestep: str) -> str:
        """Calculate calendar date for a given step"""
        days_per_step = {
            'day': 1,
            'week': 7,
            'month': 30
        }
        
        days = step * days_per_step[timestep]
        date = self.start_date + timedelta(days=days)
        return date.strftime('%Y-%m-%d')
    
    def _generate_final_predictions(self):
        """Generate final disease predictions at end of simulation"""
        print("\n📊 Generating disease predictions...")
        
        final_predictions = []
        
        for agent_name, agent in self.agents.items():
            prediction = agent.predict_disease()
            
            if isinstance(prediction, list):
                final_predictions.extend(prediction)
            else:
                final_predictions.append(prediction)
        
        # Sort by probability
        final_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display top predictions
        print("\n🎯 Top Disease Risks:")
        for i, pred in enumerate(final_predictions[:5], 1):
            print(f"   {i}. {pred['disease']}: {pred['probability']*100:.1f}% "
                  f"(~{pred['time_to_onset_years']:.0f} years)")
        
        self.final_predictions = final_predictions
    
    def _compile_results(self) -> Dict:
        """Compile all simulation results"""
        return {
            'patient_id': self.patient_id,
            'simulation_date': self.start_date.strftime('%Y-%m-%d'),
            'simulation_years': len(self.timeline) / 4 if self.timeline else 0,
            
            'current_state': self._get_current_state(),
            'trajectory': self._format_trajectory(),
            'disease_predictions': self.final_predictions,
            'interventions': self._generate_interventions(),
            
            'metadata': {
                'total_steps': len(self.timeline),
                'agents': list(self.agents.keys()),
                'data_completeness': self._calculate_data_completeness()
            }
        }
    
    def _get_current_state(self) -> Dict:
        """Get current state of all agents"""
        current = {
            'overall_health_score': self._calculate_health_score(),
            'organ_health': {}
        }
        
        for agent_name, agent in self.agents.items():
            # Simple health score based on agent state
            if agent.state.history:
                current['organ_health'][agent_name] = self._agent_health_score(agent)
        
        return current
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-10)"""
        scores = []
        
        for agent in self.agents.values():
            scores.append(self._agent_health_score(agent))
        
        return np.mean(scores) if scores else 5.0
    
    def _agent_health_score(self, agent) -> float:
        """Calculate health score for a single agent (0-10)"""
        # Get disease prediction
        prediction = agent.predict_disease()
        
        if isinstance(prediction, list):
            # Use worst prediction
            risk = max(p['probability'] for p in prediction) if prediction else 0
        else:
            risk = prediction['probability']
        
        # Convert risk to health score (inverse)
        return 10.0 * (1.0 - risk)
    
    def _format_trajectory(self) -> List[Dict]:
        """Format trajectory for output"""
        formatted = []
        
        for i, snapshot in enumerate(self.timeline):
            year = i / 4  # Quarterly snapshots
            
            formatted.append({
                'year': year,
                'date': snapshot['date'],
                'health_score': self._calculate_health_score(),
                'key_changes': self._identify_key_changes(snapshot)
            })
        
        return formatted
    
    def _identify_key_changes(self, snapshot: Dict) -> List[str]:
        """Identify significant changes in this snapshot"""
        changes = []
        
        # Check metabolic changes
        if 'metabolic' in snapshot['agents']:
            metabolic = snapshot['agents']['metabolic']
            if 'hba1c' in metabolic and metabolic['hba1c'] > 5.7:
                changes.append(f"HbA1c {metabolic['hba1c']:.1f}%")
        
        # Check cardiovascular changes
        if 'cardiovascular' in snapshot['agents']:
            cardio = snapshot['agents']['cardiovascular']
            if 'systolic_bp' in cardio and cardio['systolic_bp'] > 130:
                changes.append(f"BP {cardio['systolic_bp']:.0f}/{cardio.get('diastolic_bp', 80):.0f}")
        
        return changes[:3]  # Top 3 changes
    
    def _generate_interventions(self) -> List[Dict]:
        """Generate intervention recommendations"""
        interventions = []
        
        for prediction in self.final_predictions:
            if prediction['probability'] > 0.3:
                disease_interventions = self._get_interventions_for_disease(
                    prediction['disease'],
                    prediction['probability']
                )
                
                interventions.append({
                    'disease': prediction['disease'],
                    'current_risk': prediction['probability'],
                    'recommendations': disease_interventions
                })
        
        return interventions
    
    def _get_interventions_for_disease(self, disease: str, risk: float) -> List[Dict]:
        """Get evidence-based interventions for a disease"""
        # Evidence-based interventions from literature
        intervention_database = {
            'Type 2 Diabetes': [
                {
                    'type': 'lifestyle',
                    'intervention': '150 min/week moderate exercise',
                    'risk_reduction': 0.58,
                    'evidence': 'High (DPP study)',
                    'difficulty': 'moderate'
                },
                {
                    'type': 'lifestyle',
                    'intervention': '7% weight loss',
                    'risk_reduction': 0.58,
                    'evidence': 'High (DPP study)',
                    'difficulty': 'moderate'
                },
                {
                    'type': 'medical',
                    'intervention': 'Metformin 850mg',
                    'risk_reduction': 0.31,
                    'evidence': 'High (DPP study)',
                    'difficulty': 'low'
                }
            ],
            'Cardiovascular Disease': [
                {
                    'type': 'lifestyle',
                    'intervention': 'Mediterranean diet',
                    'risk_reduction': 0.30,
                    'evidence': 'High (PREDIMED)',
                    'difficulty': 'moderate'
                },
                {
                    'type': 'lifestyle',
                    'intervention': 'Quit smoking',
                    'risk_reduction': 0.50,
                    'evidence': 'High (Multiple RCTs)',
                    'difficulty': 'high'
                },
                {
                    'type': 'medical',
                    'intervention': 'Statin therapy',
                    'risk_reduction': 0.25,
                    'evidence': 'High (Multiple RCTs)',
                    'difficulty': 'low'
                }
            ],
            'Hypertension': [
                {
                    'type': 'lifestyle',
                    'intervention': 'DASH diet',
                    'risk_reduction': 0.40,
                    'evidence': 'High (DASH trial)',
                    'difficulty': 'moderate'
                },
                {
                    'type': 'lifestyle',
                    'intervention': 'Reduce sodium to <2g/day',
                    'risk_reduction': 0.30,
                    'evidence': 'High (Multiple studies)',
                    'difficulty': 'moderate'
                }
            ]
        }
        
        interventions = intervention_database.get(disease, [])
        
        # Calculate new risk after intervention
        for intervention in interventions:
            new_risk = risk * (1 - intervention['risk_reduction'])
            intervention['new_probability'] = new_risk
            intervention['absolute_reduction'] = risk - new_risk
        
        return interventions
    
    def _calculate_data_completeness(self) -> float:
        """Calculate what % of ideal data we have"""
        # List of ideal parameters
        ideal_params = [
            'age', 'sex', 'height', 'weight', 'bmi',
            'blood_pressure', 'fasting_glucose', 'hba1c',
            'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol',
            'creatinine', 'alt', 'ast', 'crp'
        ]
        
        available = sum(1 for param in ideal_params if param in self.patient_data)
        
        return available / len(ideal_params)
    
    def save_results(self, output_dir: str = 'outputs/simulations'):
        """Save simulation results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"simulation_{self.patient_id}_{timestamp}.json"
        filepath = output_path / filename
        
        results = self._compile_results()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        
        return filepath
