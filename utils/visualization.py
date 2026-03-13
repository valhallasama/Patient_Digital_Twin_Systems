"""
Dynamic Visualization Module
Creates interactive graphs for disease risk, timeline, and interventions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class HealthVisualization:
    """Create dynamic visualizations for patient digital twin results"""
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'diabetes': '#e74c3c',
            'cardiovascular': '#e67e22',
            'hypertension': '#f39c12',
            'ckd': '#3498db',
            'metabolic_syndrome': '#9b59b6',
            'healthy': '#2ecc71',
            'warning': '#f1c40f',
            'danger': '#e74c3c'
        }
    
    def plot_risk_timeline(
        self,
        timeline_data: List[Dict[str, Any]],
        diseases_emerged: List[Any],
        patient_id: str
    ) -> str:
        """
        Plot disease risk over time with emergence points
        Returns path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Patient {patient_id}: 5-Year Health Trajectory', fontsize=16, fontweight='bold')
        
        # Extract data
        days = [d['day'] for d in timeline_data]
        years = [d / 365 for d in days]
        
        # Plot 1: Glucose and HbA1c
        ax1 = axes[0, 0]
        glucose = [d['agent_states'].get('metabolic', {}).get('glucose', 5.0) for d in timeline_data]
        hba1c = [d['agent_states'].get('metabolic', {}).get('hba1c', 5.0) for d in timeline_data]
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(years, glucose, 'b-', label='Glucose', linewidth=2)
        line2 = ax1_twin.plot(years, hba1c, 'r-', label='HbA1c', linewidth=2)
        
        ax1.axhline(y=6.5, color='orange', linestyle='--', alpha=0.5, label='Prediabetes threshold')
        ax1_twin.axhline(y=6.5, color='red', linestyle='--', alpha=0.5, label='Diabetes threshold')
        
        ax1.set_xlabel('Years', fontsize=12)
        ax1.set_ylabel('Glucose (mmol/L)', color='b', fontsize=12)
        ax1_twin.set_ylabel('HbA1c (%)', color='r', fontsize=12)
        ax1.set_title('Metabolic Health Trajectory', fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Blood Pressure
        ax2 = axes[0, 1]
        bp_sys = [d['agent_states'].get('cardiovascular', {}).get('systolic_bp', 120) for d in timeline_data]
        bp_dia = [d['agent_states'].get('cardiovascular', {}).get('diastolic_bp', 80) for d in timeline_data]
        
        ax2.plot(years, bp_sys, 'r-', label='Systolic', linewidth=2)
        ax2.plot(years, bp_dia, 'b-', label='Diastolic', linewidth=2)
        ax2.axhline(y=140, color='red', linestyle='--', alpha=0.5, label='Hypertension threshold')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Years', fontsize=12)
        ax2.set_ylabel('Blood Pressure (mmHg)', fontsize=12)
        ax2.set_title('Cardiovascular Health Trajectory', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Kidney Function
        ax3 = axes[1, 0]
        egfr = [d['agent_states'].get('renal', {}).get('egfr', 100) for d in timeline_data]
        
        ax3.plot(years, egfr, 'g-', linewidth=2)
        ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='CKD Stage 3')
        ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='CKD Stage 4')
        ax3.fill_between(years, 90, 120, alpha=0.2, color='green', label='Normal range')
        
        ax3.set_xlabel('Years', fontsize=12)
        ax3.set_ylabel('eGFR (mL/min/1.73m²)', fontsize=12)
        ax3.set_title('Kidney Function Trajectory', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Agent Stress Levels
        ax4 = axes[1, 1]
        agent_names = ['Cardiovascular', 'Metabolic', 'Renal', 'Hepatic', 'Immune', 'Endocrine', 'Neural']
        
        # Get final stress levels
        if timeline_data:
            final_state = timeline_data[-1]['agent_states']
            stress_levels = []
            for agent in ['cardiovascular', 'metabolic', 'renal', 'hepatic', 'immune', 'endocrine', 'neural']:
                stress = final_state.get(agent, {}).get('stress_level', 0)
                stress_levels.append(stress * 100)
            
            colors_bar = ['red' if s > 70 else 'orange' if s > 40 else 'green' for s in stress_levels]
            bars = ax4.barh(agent_names, stress_levels, color=colors_bar, alpha=0.7)
            
            ax4.set_xlabel('Stress Level (%)', fontsize=12)
            ax4.set_title('Final Agent Stress Levels', fontsize=14, fontweight='bold')
            ax4.set_xlim(0, 100)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, stress_levels)):
                ax4.text(val + 2, i, f'{val:.0f}%', va='center', fontsize=10)
        
        # Mark disease emergence points on all plots
        for disease in diseases_emerged:
            day = disease.day_emerged
            year = day / 365
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=year, color='red', linestyle=':', alpha=0.5, linewidth=2)
                ax.text(year, ax.get_ylim()[1] * 0.95, disease.name.split()[0], 
                       rotation=90, va='top', ha='right', fontsize=9, color='red')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'{patient_id}_timeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_intervention_impact(
        self,
        disease: str,
        current_risk: float,
        interventions_data: List[Dict[str, Any]],
        patient_id: str
    ) -> str:
        """
        Plot impact of different interventions on disease risk
        Returns path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Intervention Impact Analysis: {disease.title()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Risk Reduction Bar Chart
        if interventions_data:
            intervention_names = []
            risk_reductions = []
            new_risks = []
            
            for item in interventions_data[:5]:  # Top 5
                intervention = item['intervention']
                intervention_names.append(intervention.description[:40])
                risk_reductions.append(intervention.risk_reduction * 100)
                new_risk = current_risk * (1 - intervention.risk_reduction)
                new_risks.append(new_risk * 100)
            
            x = np.arange(len(intervention_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, [current_risk * 100] * len(intervention_names), 
                           width, label='Current Risk', color='red', alpha=0.6)
            bars2 = ax1.bar(x + width/2, new_risks, width, label='Risk After Intervention', 
                           color='green', alpha=0.6)
            
            ax1.set_ylabel('Risk (%)', fontsize=12)
            ax1.set_title('Risk Before vs After Interventions', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(intervention_names, rotation=45, ha='right', fontsize=9)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                reduction = risk_reductions[i]
                height = bar2.get_height()
                ax1.text(bar2.get_x() + bar2.get_width()/2., height,
                        f'-{reduction:.0f}%', ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        
        # Plot 2: Combined Intervention Effect
        if len(interventions_data) >= 3:
            scenarios = ['No intervention', '1 intervention', '2 interventions', '3 interventions']
            risks = [current_risk * 100]
            
            # Calculate cumulative effect
            cumulative_reduction = 1.0
            for i in range(min(3, len(interventions_data))):
                intervention = interventions_data[i]['intervention']
                cumulative_reduction *= (1 - intervention.risk_reduction)
                risks.append(current_risk * cumulative_reduction * 100)
            
            colors_scenario = ['red', 'orange', 'yellow', 'green'][:len(risks)]
            bars = ax2.bar(scenarios[:len(risks)], risks, color=colors_scenario, alpha=0.7)
            
            ax2.set_ylabel('Disease Risk (%)', fontsize=12)
            ax2.set_title('Cumulative Effect of Multiple Interventions', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, risk in zip(bars, risks):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{risk:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add target line
            ax2.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target risk (<10%)')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'{patient_id}_interventions_{disease}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def plot_lifestyle_comparison(
        self,
        current_lifestyle: Dict[str, Any],
        recommended_lifestyle: Dict[str, Any],
        patient_id: str
    ) -> str:
        """
        Plot current vs recommended lifestyle
        Returns path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Lifestyle Improvement Plan', fontsize=16, fontweight='bold')
        
        categories = ['Exercise\n(sessions/week)', 'Sleep\n(hours/night)', 'Diet Quality\n(0-10)', 'Stress Level\n(0-10)']
        
        # Normalize values to 0-10 scale
        current_values = [
            current_lifestyle.get('exercise_sessions_per_week', 1),
            current_lifestyle.get('sleep_hours', 6.5),
            {'poor': 3, 'moderate': 6, 'good': 9}.get(current_lifestyle.get('diet_quality', 'moderate'), 6),
            {'low': 3, 'moderate': 6, 'high': 9}.get(current_lifestyle.get('stress_level', 'moderate'), 6)
        ]
        
        recommended_values = [
            recommended_lifestyle.get('exercise_sessions_per_week', 5),
            recommended_lifestyle.get('sleep_hours', 7.5),
            {'poor': 3, 'moderate': 6, 'good': 9}.get(recommended_lifestyle.get('diet_quality', 'good'), 9),
            {'low': 3, 'moderate': 6, 'high': 9}.get(recommended_lifestyle.get('stress_level', 'low'), 3)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_values, width, label='Current', color='orange', alpha=0.7)
        bars2 = ax.bar(x + width/2, recommended_values, width, label='Recommended', color='green', alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Current vs Recommended Lifestyle', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and improvement arrows
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{height1:.1f}', ha='center', va='bottom', fontsize=10)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{height2:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Draw improvement arrow
            if height2 > height1:
                improvement = ((height2 - height1) / height1) * 100
                ax.annotate('', xy=(i, height2), xytext=(i, height1),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax.text(i, (height1 + height2) / 2, f'+{improvement:.0f}%',
                       ha='center', fontsize=9, color='green', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'{patient_id}_lifestyle_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_summary_dashboard(
        self,
        patient_id: str,
        current_risks: Dict[str, float],
        predicted_risks: Dict[str, float],
        timeline_data: List[Dict[str, Any]],
        diseases_emerged: List[Any]
    ) -> str:
        """
        Create comprehensive dashboard with all key metrics
        Returns path to saved figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Patient {patient_id}: Comprehensive Health Dashboard', fontsize=18, fontweight='bold')
        
        # Risk gauge charts
        for i, (disease, risk) in enumerate(current_risks.items()):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col], projection='polar')
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            r = np.ones(100)
            
            # Color zones
            ax.fill_between(theta[:33], 0, 1, color='green', alpha=0.3, label='Low risk')
            ax.fill_between(theta[33:66], 0, 1, color='yellow', alpha=0.3, label='Medium risk')
            ax.fill_between(theta[66:], 0, 1, color='red', alpha=0.3, label='High risk')
            
            # Risk needle
            risk_angle = risk * np.pi
            ax.plot([risk_angle, risk_angle], [0, 0.8], 'k-', linewidth=3)
            ax.plot(risk_angle, 0.8, 'ko', markersize=10)
            
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{disease.replace("_", " ").title()}\n{risk:.1%} Risk', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.spines['polar'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'{patient_id}_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


# Global instance
_visualizer = None

def get_visualizer() -> HealthVisualization:
    """Get or create global visualizer"""
    global _visualizer
    if _visualizer is None:
        _visualizer = HealthVisualization()
    return _visualizer
