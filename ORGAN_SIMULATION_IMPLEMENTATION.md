# Data-Driven Organ Simulation System

## Overview

This document outlines the implementation of a **personalized, explainable organ simulation system** that creates a digital twin of individual patients and simulates their health trajectory based on learned dynamics from 135K NHANES patients.

## Core Philosophy

**NOT Acceptable:**
- ❌ "You have 90% chance of fatty liver based on population statistics"
- ❌ Black-box predictions without mechanistic explanation
- ❌ Hand-coded simulation parameters
- ❌ One-size-fits-all recommendations

**What We Build:**
- ✅ "Your liver ALT rose from 45→72 over 6 months due to alcohol + metabolic stress"
- ✅ Mechanistic, explainable predictions
- ✅ Dynamics learned from real patient data
- ✅ Personalized to individual's specific organ states and lifestyle

---

## Architecture

### 1. Patient Input Processing

**Input Format:**
```python
patient_profile = {
    'demographics': {
        'age': 40,
        'sex': 'male',
        'bmi': 28.5
    },
    'organ_biomarkers': {
        'liver': {'ALT': 45, 'AST': 38, 'bilirubin': 0.8},
        'metabolic': {'glucose': 110, 'HbA1c': 5.9, 'insulin': 15, 'triglycerides': 180},
        'cardiovascular': {'systolic_bp': 135, 'diastolic_bp': 85, 'cholesterol': 220},
        'kidney': {'creatinine': 1.1, 'BUN': 18},
        'immune': {'WBC': 7.5},
        'neural': {'cognitive_score': 0.85}
    },
    'lifestyle': {
        'exercise_frequency': 0.2,  # 0-1 scale (rarely = 0.2)
        'alcohol_consumption': 0.9,  # 0-1 scale (heavy = 0.9)
        'diet_quality': 0.3,  # 0-1 scale (poor = 0.3)
        'sleep_hours': 5.5,
        'smoking': 0.0
    },
    'medications': ['none'],
    'medical_history': ['fatty_liver']
}
```

### 2. Digital Twin Initialization

```python
class DigitalTwin:
    """Personalized digital representation of patient's organ systems"""
    
    def __init__(self, patient_profile, pretrained_model):
        self.demographics = patient_profile['demographics']
        self.current_organs = self._normalize_biomarkers(
            patient_profile['organ_biomarkers']
        )
        self.lifestyle = patient_profile['lifestyle']
        self.medications = patient_profile['medications']
        
        # Use pretrained GNN + Transformer
        self.gnn = pretrained_model.gnn
        self.transformer = pretrained_model.transformer
        
        # Dynamics predictor (to be trained)
        self.dynamics_net = OrganDynamicsPredictor(
            gnn_dim=64,
            transformer_dim=512,
            organ_dims=self._get_organ_dims()
        )
    
    def _normalize_biomarkers(self, biomarkers):
        """Convert raw lab values to normalized features"""
        # Use same normalization as training data
        # Returns organ feature tensors matching model input format
```

### 3. Learned Organ Dynamics

**Key Innovation: Learn from Real Patient Transitions**

```python
class OrganDynamicsPredictor(nn.Module):
    """
    Learns organ state evolution from NHANES temporal data
    
    Training: Given (state[t], lifestyle[t]) → predict state[t+1]
    Uses real patient transitions, not hand-coded rules
    """
    
    def __init__(self, gnn_dim, transformer_dim, organ_dims):
        super().__init__()
        
        # Combine GNN organ interactions + Transformer temporal context
        self.fusion = nn.Sequential(
            nn.Linear(gnn_dim * 7 + transformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Predict delta for each organ system
        self.delta_predictors = nn.ModuleDict({
            organ: nn.Sequential(
                nn.Linear(256 + lifestyle_dim, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )
            for organ, dim in organ_dims.items()
        })
        
    def forward(self, organ_states, lifestyle, gnn_context, temporal_context):
        """
        Predict how organs change in next timestep
        
        Args:
            organ_states: Current organ biomarkers
            lifestyle: Current lifestyle factors
            gnn_context: GNN embedding (organ interactions)
            temporal_context: Transformer embedding (temporal patterns)
        
        Returns:
            delta_organs: Predicted change in each organ
        """
        # Fuse multi-modal context
        combined = torch.cat([gnn_context, temporal_context], dim=-1)
        shared_features = self.fusion(combined)
        
        # Predict organ-specific deltas
        deltas = {}
        for organ, predictor in self.delta_predictors.items():
            organ_input = torch.cat([shared_features, lifestyle], dim=-1)
            deltas[organ] = predictor(organ_input)
        
        return deltas
```

### 4. Forward Simulation Engine

```python
class OrganSimulator:
    """Simulate patient's health trajectory over time"""
    
    def __init__(self, digital_twin):
        self.twin = digital_twin
        self.history = []
        
    def simulate_forward(
        self,
        months: int,
        lifestyle_scenario: Dict[str, float],
        intervention_start_month: int = 0
    ):
        """
        Simulate organ evolution month-by-month
        
        Args:
            months: Simulation horizon (e.g., 24 months)
            lifestyle_scenario: Lifestyle parameters to use
            intervention_start_month: When to apply lifestyle changes
        
        Returns:
            trajectory: Monthly organ states
            events: Detected disease onsets with explanations
        """
        trajectory = []
        events = []
        current_state = self.twin.current_organs.copy()
        
        for month in range(months):
            # Apply intervention if applicable
            if month >= intervention_start_month:
                lifestyle = lifestyle_scenario
            else:
                lifestyle = self.twin.lifestyle
            
            # Get GNN context (organ interactions)
            gnn_emb = self.twin.gnn(
                current_state,
                edge_index=self.twin.gnn.edge_index,
                return_hidden=True
            )
            
            # Get Transformer context (temporal patterns)
            # Use last 12 months of history
            history_window = trajectory[-12:] if len(trajectory) >= 12 else trajectory
            if history_window:
                temporal_emb = self.twin.transformer(
                    torch.stack(history_window),
                    time_deltas=torch.ones(len(history_window))
                )
            else:
                temporal_emb = torch.zeros(512)  # Initial state
            
            # Predict organ changes (LEARNED from data)
            deltas = self.twin.dynamics_net(
                current_state,
                lifestyle,
                gnn_emb,
                temporal_emb
            )
            
            # Update organ states
            next_state = {
                organ: current_state[organ] + deltas[organ]
                for organ in current_state.keys()
            }
            
            # Check for disease thresholds
            disease_events = self._check_disease_onset(
                current_state,
                next_state,
                month
            )
            events.extend(disease_events)
            
            # Record trajectory
            trajectory.append({
                'month': month,
                'organs': next_state.copy(),
                'lifestyle': lifestyle.copy(),
                'events': disease_events
            })
            
            current_state = next_state
        
        return trajectory, events
    
    def _check_disease_onset(self, prev_state, curr_state, month):
        """
        Detect disease onset based on clinical thresholds
        Generate mechanistic explanations
        """
        events = []
        
        # Example: Fatty liver detection
        liver_prev = prev_state['liver']
        liver_curr = curr_state['liver']
        
        if (liver_curr['ALT'] > 70 and 
            liver_prev['ALT'] <= 70):
            
            # Generate mechanistic explanation
            alt_increase = liver_curr['ALT'] - liver_prev['ALT']
            
            # Identify contributing factors
            factors = []
            if self.twin.lifestyle['alcohol_consumption'] > 0.7:
                factors.append("high alcohol consumption")
            if curr_state['metabolic']['glucose'] > 100:
                factors.append("elevated glucose (metabolic stress)")
            if self.twin.lifestyle['exercise_frequency'] < 0.3:
                factors.append("insufficient exercise")
            
            explanation = (
                f"Fatty liver detected at month {month}. "
                f"ALT elevated to {liver_curr['ALT']:.1f} (normal <40). "
                f"Contributing factors: {', '.join(factors)}. "
                f"Liver enzymes rose {alt_increase:.1f} points due to sustained hepatic stress."
            )
            
            events.append({
                'disease': 'fatty_liver',
                'month': month,
                'severity': 'moderate' if liver_curr['ALT'] < 80 else 'severe',
                'explanation': explanation,
                'biomarkers': {
                    'ALT': liver_curr['ALT'],
                    'AST': liver_curr['AST']
                }
            })
        
        # Add similar checks for other diseases...
        
        return events
```

### 5. Intervention Analysis

```python
class InterventionAnalyzer:
    """Compare different lifestyle/treatment scenarios"""
    
    def analyze_scenarios(self, digital_twin, scenarios, months=24):
        """
        Simulate multiple futures and compare outcomes
        
        Args:
            digital_twin: Patient's digital twin
            scenarios: Dict of {scenario_name: lifestyle_params}
            months: Simulation horizon
        
        Returns:
            comparison: Side-by-side trajectory comparison
        """
        results = {}
        
        for name, lifestyle in scenarios.items():
            simulator = OrganSimulator(digital_twin)
            trajectory, events = simulator.simulate_forward(
                months=months,
                lifestyle_scenario=lifestyle
            )
            
            results[name] = {
                'trajectory': trajectory,
                'events': events,
                'final_state': trajectory[-1]['organs'],
                'disease_free_months': self._count_disease_free_months(events, months)
            }
        
        return self._generate_comparison_report(results)
    
    def _generate_comparison_report(self, results):
        """Generate human-readable comparison"""
        report = []
        
        for scenario, data in results.items():
            report.append(f"\n## Scenario: {scenario}")
            report.append(f"Disease-free months: {data['disease_free_months']}/{len(data['trajectory'])}")
            
            if data['events']:
                report.append("\nDisease events:")
                for event in data['events']:
                    report.append(f"  - Month {event['month']}: {event['disease']}")
                    report.append(f"    {event['explanation']}")
            else:
                report.append("\nNo disease events detected ✓")
            
            # Show key biomarker trends
            report.append("\nKey biomarker changes:")
            initial = data['trajectory'][0]['organs']
            final = data['final_state']
            
            for organ in ['liver', 'metabolic', 'cardiovascular']:
                for marker in initial[organ].keys():
                    change = final[organ][marker] - initial[organ][marker]
                    direction = "↑" if change > 0 else "↓"
                    report.append(
                        f"  {organ}.{marker}: "
                        f"{initial[organ][marker]:.1f} → {final[organ][marker]:.1f} "
                        f"({direction}{abs(change):.1f})"
                    )
        
        return "\n".join(report)
```

### 6. Training the Dynamics Predictor

```python
def train_dynamics_predictor(nhanes_data, pretrained_model, epochs=50):
    """
    Train dynamics predictor on real patient transitions
    
    Uses NHANES patients with multiple survey cycles to learn:
    "Given current state + lifestyle, what happens next?"
    """
    
    dynamics_net = OrganDynamicsPredictor(...)
    optimizer = torch.optim.AdamW(dynamics_net.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for patient in nhanes_data:
            # Patient has multiple timepoints
            for t in range(len(patient.trajectory) - 1):
                current_state = patient.organs[t]
                lifestyle = patient.lifestyle[t]
                actual_next_state = patient.organs[t + 1]
                
                # Get pretrained embeddings
                gnn_emb = pretrained_model.gnn(current_state)
                temporal_emb = pretrained_model.transformer(
                    patient.organs[:t+1]
                )
                
                # Predict next state
                predicted_deltas = dynamics_net(
                    current_state,
                    lifestyle,
                    gnn_emb,
                    temporal_emb
                )
                
                predicted_next = {
                    organ: current_state[organ] + predicted_deltas[organ]
                    for organ in current_state.keys()
                }
                
                # Loss: How well do we predict actual transitions?
                loss = sum([
                    F.mse_loss(predicted_next[organ], actual_next_state[organ])
                    for organ in current_state.keys()
                ])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"Epoch {epoch}: Transition prediction loss = {loss.item():.4f}")
```

---

## Example Usage

```python
# 1. Load pretrained model
model = torch.load('./models/finetuned/finetuned_best.pt')

# 2. Train dynamics predictor
dynamics_net = train_dynamics_predictor(nhanes_data, model)

# 3. Create patient digital twin
patient = {
    'demographics': {'age': 40, 'sex': 'male', 'bmi': 28.5},
    'organ_biomarkers': {
        'liver': {'ALT': 45, 'AST': 38},
        'metabolic': {'glucose': 110, 'HbA1c': 5.9},
        # ... other organs
    },
    'lifestyle': {
        'exercise_frequency': 0.2,
        'alcohol_consumption': 0.9,
        'diet_quality': 0.3
    }
}

twin = DigitalTwin(patient, model)
twin.dynamics_net = dynamics_net

# 4. Define intervention scenarios
scenarios = {
    'current_behavior': patient['lifestyle'],
    'moderate_improvement': {
        'exercise_frequency': 0.5,
        'alcohol_consumption': 0.5,
        'diet_quality': 0.6
    },
    'aggressive_intervention': {
        'exercise_frequency': 0.8,
        'alcohol_consumption': 0.1,
        'diet_quality': 0.8
    }
}

# 5. Simulate and compare
analyzer = InterventionAnalyzer()
comparison = analyzer.analyze_scenarios(twin, scenarios, months=24)
print(comparison)
```

**Expected Output:**
```
## Scenario: current_behavior
Disease-free months: 4/24
Disease events:
  - Month 6: fatty_liver
    Fatty liver detected at month 6. ALT elevated to 72.3 (normal <40).
    Contributing factors: high alcohol consumption, elevated glucose, insufficient exercise.
    Liver enzymes rose 27.3 points due to sustained hepatic stress.

Key biomarker changes:
  liver.ALT: 45.0 → 78.5 (↑33.5)
  metabolic.glucose: 110.0 → 125.3 (↑15.3)

## Scenario: moderate_improvement
Disease-free months: 24/24 ✓
No disease events detected

Key biomarker changes:
  liver.ALT: 45.0 → 38.2 (↓6.8)
  metabolic.glucose: 110.0 → 98.5 (↓11.5)
```

---

## Implementation Timeline

1. **After Stage 2 completes:** Extract pretrained GNN + Transformer
2. **Prepare training data:** Extract NHANES temporal transitions (patients with multiple cycles)
3. **Train dynamics predictor:** Learn state evolution from real data (~2-3 hours)
4. **Implement simulation engine:** Forward simulation with learned dynamics
5. **Add disease detection:** Clinical threshold-based detection with explanations
6. **Build intervention analyzer:** Scenario comparison system
7. **Test with examples:** Validate on real patient cases

---

## Key Advantages

1. **Data-Driven:** All dynamics learned from 135K real patients, not hand-coded
2. **Personalized:** Simulates individual's specific organ states and lifestyle
3. **Explainable:** Every prediction has mechanistic evidence
4. **Actionable:** Shows which interventions prevent disease progression
5. **Transparent:** User sees exact biomarker trajectories, not black-box probabilities
