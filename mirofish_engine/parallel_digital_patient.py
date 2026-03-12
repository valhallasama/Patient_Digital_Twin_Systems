"""
Parallel Digital Patient - MiroFish-Inspired Simulation Engine
High-fidelity physiological simulation with swarm intelligence
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import json
from pathlib import Path

from .organ_agents import create_agent_from_seed
from .internal_milieu import InternalMilieu
from .body_system_agent import BodySystemAgent


class DiseaseEmergence:
    """Detected disease emergence from agent interactions"""
    
    def __init__(self, name: str, probability: float, day_emerged: int,
                 causative_agents: List[str], mechanism: str):
        self.name = name
        self.probability = probability
        self.day_emerged = day_emerged
        self.causative_agents = causative_agents
        self.mechanism = mechanism
    
    def to_dict(self):
        return {
            'name': self.name,
            'probability': self.probability,
            'day_emerged': self.day_emerged,
            'causative_agents': self.causative_agents,
            'mechanism': self.mechanism
        }


class ParallelDigitalPatient:
    """
    High-fidelity parallel digital patient simulation
    Inspired by MiroFish's parallel digital world
    """
    
    def __init__(self, patient_id: str, seed_information: Dict[str, Any]):
        self.patient_id = patient_id
        self.seed_information = seed_information
        
        # Initialize agents from seed information
        self.agents: Dict[str, BodySystemAgent] = self._initialize_agents()
        
        # Initialize environment
        self.environment = InternalMilieu(
            initial_composition=seed_information.get('initial_composition', {})
        )
        
        # Connect agents to environment
        for agent in self.agents.values():
            agent.environment = self.environment
        
        # Simulation state
        self.timeline = []
        self.diseases_emerged = []
        self.current_day = 0
    
    def _initialize_agents(self) -> Dict[str, BodySystemAgent]:
        """Initialize all body system agents from seed information"""
        agents = {}
        
        agent_seeds = self.seed_information.get('agent_seeds', {})
        
        # Create each agent type
        for agent_type in ['cardiovascular', 'metabolic', 'renal', 'hepatic', 
                          'immune', 'endocrine', 'neural']:
            seed_data = agent_seeds.get(agent_type, {})
            agents[agent_type] = create_agent_from_seed(agent_type, seed_data)
        
        return agents
    
    def simulate_future(self, days: int = 1825, interventions: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Simulate patient's future physiological evolution
        
        Args:
            days: Number of days to simulate (default 5 years)
            interventions: List of interventions to apply at specific days
        
        Returns:
            Timeline of simulation states
        """
        print(f"\n{'='*80}")
        print(f"PARALLEL DIGITAL PATIENT SIMULATION")
        print(f"{'='*80}")
        print(f"Patient ID: {self.patient_id}")
        print(f"Simulation duration: {days} days ({days/365:.1f} years)")
        print(f"Agents: {len(self.agents)}")
        print(f"{'='*80}\n")
        
        interventions = interventions or []
        intervention_schedule = {i['day']: i for i in interventions}
        
        for day in range(days):
            # Apply scheduled interventions
            if day in intervention_schedule:
                self.environment.apply_intervention(intervention_schedule[day])
            
            # Simulate one day
            day_state = self._simulate_one_day()
            
            # Record state
            self.timeline.append(day_state)
            
            # Check for disease emergence
            emerged_diseases = self._detect_disease_emergence()
            if emerged_diseases:
                for disease in emerged_diseases:
                    if disease.name not in [d.name for d in self.diseases_emerged]:
                        self.diseases_emerged.append(disease)
                        print(f"⚠️  Day {day}: {disease.name} emerged (probability: {disease.probability:.1%})")
            
            # Progress indicator
            if day % 365 == 0 and day > 0:
                print(f"✓ Year {day//365} complete")
        
        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total diseases emerged: {len(self.diseases_emerged)}")
        for disease in self.diseases_emerged:
            print(f"  - {disease.name}: {disease.probability:.1%} (Day {disease.day_emerged})")
        print(f"{'='*80}\n")
        
        return self.timeline
    
    def _simulate_one_day(self) -> Dict[str, Any]:
        """Simulate one day of physiological activity"""
        # 1. Each agent perceives environment
        perceptions = {
            name: agent.perceive(self.environment)
            for name, agent in self.agents.items()
        }
        
        # 2. Each agent decides action
        decisions = {
            name: agent.decide(perceptions[name])
            for name, agent in self.agents.items()
        }
        
        # 3. Agents interact with each other (swarm intelligence!)
        interactions = []
        for agent in self.agents.values():
            other_agents = [a for a in self.agents.values() if a != agent]
            agent_interactions = agent.interact(other_agents)
            interactions.extend(agent_interactions)
        
        # 4. Update environment based on all decisions and interactions
        self.environment.update(decisions, interactions)
        
        # 5. Agents execute actions and update their states
        for name, agent in self.agents.items():
            agent.act(decisions[name])
            agent.age_one_day()
        
        # 6. Record snapshot
        snapshot = {
            'day': self.current_day,
            'timestamp': datetime.now(),
            'environment': self.environment.get_state(),
            'agents': {name: agent.to_dict() for name, agent in self.agents.items()},
            'interactions_count': len(interactions)
        }
        
        self.current_day += 1
        
        return snapshot
    
    def _detect_disease_emergence(self) -> List[DiseaseEmergence]:
        """
        Detect disease emergence from agent interactions
        This is where swarm intelligence reveals disease patterns
        """
        diseases = []
        
        # Get current agent states
        metabolic = self.agents['metabolic']
        cardiovascular = self.agents['cardiovascular']
        renal = self.agents['renal']
        hepatic = self.agents['hepatic']
        immune = self.agents['immune']
        
        # Type 2 Diabetes detection
        if (metabolic.state['hba1c'] > 6.5 and
            metabolic.state['insulin_resistance'] > 0.6 and
            metabolic.state['beta_cell_function'] < 0.7):
            
            probability = min(0.95, 
                0.5 + metabolic.state['insulin_resistance'] * 0.5 +
                (1.0 - metabolic.state['beta_cell_function']) * 0.3)
            
            diseases.append(DiseaseEmergence(
                name='Type 2 Diabetes',
                probability=probability,
                day_emerged=self.current_day,
                causative_agents=['metabolic', 'endocrine', 'immune'],
                mechanism='Insulin resistance + beta-cell dysfunction + inflammation'
            ))
        
        # Cardiovascular Disease detection
        if (cardiovascular.state['systolic_bp'] > 140 and
            hepatic.state['ldl'] > 4.0 and
            cardiovascular.state['atherosclerosis_level'] > 0.3):
            
            probability = min(0.90,
                0.3 + (cardiovascular.state['systolic_bp'] - 120) / 100 +
                (hepatic.state['ldl'] - 3.0) / 10 +
                cardiovascular.state['atherosclerosis_level'])
            
            diseases.append(DiseaseEmergence(
                name='Cardiovascular Disease',
                probability=probability,
                day_emerged=self.current_day,
                causative_agents=['cardiovascular', 'hepatic', 'immune'],
                mechanism='Hypertension + dyslipidemia + atherosclerosis + inflammation'
            ))
        
        # Chronic Kidney Disease detection
        if (renal.state['egfr'] < 60 and
            renal.state['damage_level'] > 0.3):
            
            probability = min(0.85,
                0.4 + (100 - renal.state['egfr']) / 100 +
                renal.state['damage_level'])
            
            diseases.append(DiseaseEmergence(
                name='Chronic Kidney Disease Stage 3',
                probability=probability,
                day_emerged=self.current_day,
                causative_agents=['renal', 'cardiovascular', 'metabolic'],
                mechanism='Hypertensive + diabetic nephropathy'
            ))
        
        # Metabolic Syndrome detection
        if (metabolic.state['insulin_resistance'] > 0.5 and
            cardiovascular.state['systolic_bp'] > 130 and
            hepatic.state['ldl'] > 3.5 and
            immune.state['inflammation'] > 0.4):
            
            probability = 0.75
            
            diseases.append(DiseaseEmergence(
                name='Metabolic Syndrome',
                probability=probability,
                day_emerged=self.current_day,
                causative_agents=['metabolic', 'cardiovascular', 'hepatic', 'immune'],
                mechanism='Insulin resistance + hypertension + dyslipidemia + inflammation'
            ))
        
        return diseases
    
    def trace_disease_pathway(self, disease_name: str) -> Dict[str, Any]:
        """Trace how agent interactions led to disease emergence"""
        disease = next((d for d in self.diseases_emerged if d.name == disease_name), None)
        if not disease:
            return {'error': f'Disease {disease_name} not found'}
        
        # Analyze timeline leading up to disease emergence
        emergence_day = disease.day_emerged
        lookback_days = min(365, emergence_day)  # Look back up to 1 year
        
        relevant_timeline = self.timeline[emergence_day - lookback_days:emergence_day]
        
        # Extract key events
        key_events = []
        for i, snapshot in enumerate(relevant_timeline):
            day = snapshot['day']
            agents_state = snapshot['agents']
            
            # Check for significant changes
            for agent_name in disease.causative_agents:
                if agent_name in agents_state:
                    agent_data = agents_state[agent_name]
                    if agent_data['stress_level'] > 0.5:
                        key_events.append({
                            'day': day,
                            'event': f'{agent_name} agent stressed',
                            'stress_level': agent_data['stress_level']
                        })
        
        return {
            'disease': disease.name,
            'emergence_day': emergence_day,
            'causative_agents': disease.causative_agents,
            'mechanism': disease.mechanism,
            'key_events': key_events[-10:],  # Last 10 events
            'probability': disease.probability
        }
    
    def chat_with_agent(self, agent_name: str, question: str) -> str:
        """Chat with a specific body system agent"""
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
        
        agent = self.agents[agent_name]
        context = {
            'current_state': agent.state,
            'recent_memory': agent.memory[-10:],
            'personality': agent.personality,
            'environment': self.environment.get_state()
        }
        
        return agent.respond_to_query(question, context)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive prediction report"""
        return {
            'patient_id': self.patient_id,
            'simulation_days': self.current_day,
            'simulation_years': self.current_day / 365,
            
            'diseases_predicted': [d.to_dict() for d in self.diseases_emerged],
            
            'agent_final_states': {
                name: agent.to_dict()
                for name, agent in self.agents.items()
            },
            
            'environment_final_state': self.environment.get_state(),
            
            'disease_pathways': {
                disease.name: self.trace_disease_pathway(disease.name)
                for disease in self.diseases_emerged
            },
            
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary"""
        summary = f"Parallel Digital Patient Simulation Report\n"
        summary += f"Patient ID: {self.patient_id}\n"
        summary += f"Simulation: {self.current_day} days ({self.current_day/365:.1f} years)\n\n"
        
        if self.diseases_emerged:
            summary += "Diseases Predicted to Emerge:\n"
            for disease in self.diseases_emerged:
                summary += f"  - {disease.name}: {disease.probability:.1%} "
                summary += f"(Day {disease.day_emerged}, ~{disease.day_emerged/365:.1f} years)\n"
                summary += f"    Mechanism: {disease.mechanism}\n"
        else:
            summary += "No major diseases predicted to emerge in simulation period.\n"
        
        summary += "\nFinal Agent States:\n"
        for name, agent in self.agents.items():
            summary += f"  - {name.capitalize()}: {agent.health_status.value} "
            summary += f"(stress: {agent.stress_level:.1%})\n"
        
        return summary
    
    def save_results(self, output_dir: str = "outputs/mirofish_simulations"):
        """Save simulation results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.patient_id}_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(output_path / filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {output_path / filename}")
        
        return output_path / filename
