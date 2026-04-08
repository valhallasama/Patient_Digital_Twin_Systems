#!/usr/bin/env python3
"""
Organ System Registry

Scientifically honest classification of which organs are learned from temporal data
vs which are placeholders awaiting longitudinal cohort data.

This is critical for research integrity and publishability.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class LearningStatus(Enum):
    """Classification of how organ dynamics are determined"""
    
    TEMPORAL_LEARNED = "learned_from_longitudinal_trajectories"
    # Organ state transitions learned from real patient data over time
    # Example: glucose_t+1 = f(glucose_t, lifestyle_t) trained on 33,994 transitions
    
    PLACEHOLDER = "awaiting_longitudinal_cohort_data"
    # Organ has constant values in current dataset (NHANES)
    # Cannot learn temporal dynamics until longitudinal data acquired
    # Example: ALT constant at 28 for all patients - data quality issue
    
    CROSS_SECTIONAL = "cross_sectional_correlations_only"
    # Can estimate current state from demographics/risk factors
    # But cannot predict temporal trajectories
    # Example: HDL inferred from age/gender, but no trajectory learning


@dataclass
class OrganSystemInfo:
    """Information about an organ system's learning status"""
    name: str
    parameters: List[str]
    learning_status: LearningStatus
    data_source: str
    sample_size: Optional[int]
    temporal_coverage: Optional[str]
    notes: str


class OrganSystemRegistry:
    """
    Registry of all organ systems with honest assessment of learning status
    
    This is the source of truth for what is actually learned vs placeholder.
    """
    
    def __init__(self):
        self.organs = self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[str, OrganSystemInfo]:
        """Initialize registry with current status"""
        
        return {
            'metabolic': OrganSystemInfo(
                name='Metabolic System',
                parameters=['glucose', 'HbA1c', 'insulin', 'triglycerides'],
                learning_status=LearningStatus.TEMPORAL_LEARNED,
                data_source='NHANES temporal transitions',
                sample_size=33994,
                temporal_coverage='2-cycle transitions (typically 2-4 years apart)',
                notes='✅ TRUE DIGITAL TWIN: Learned from real patient trajectories. '
                      'Model predicts glucose_t+1 from glucose_t, lifestyle_t, age_t. '
                      'Validated on held-out patients.'
            ),
            
            'cardiovascular': OrganSystemInfo(
                name='Cardiovascular System',
                parameters=['systolic_bp', 'diastolic_bp', 'total_cholesterol', 'HDL', 'LDL'],
                learning_status=LearningStatus.TEMPORAL_LEARNED,
                data_source='NHANES temporal transitions',
                sample_size=33994,
                temporal_coverage='2-cycle transitions (typically 2-4 years apart)',
                notes='✅ TRUE DIGITAL TWIN: BP trajectories learned from real patients. '
                      'HDL has limited variation but BP dynamics are well-learned. '
                      'Model predicts BP_t+1 from BP_t, lifestyle_t, age_t.'
            ),
            
            'kidney': OrganSystemInfo(
                name='Kidney System',
                parameters=['creatinine', 'BUN'],
                learning_status=LearningStatus.TEMPORAL_LEARNED,
                data_source='NHANES temporal transitions',
                sample_size=33994,
                temporal_coverage='2-cycle transitions (typically 2-4 years apart)',
                notes='✅ TRUE DIGITAL TWIN: Creatinine trajectories learned from real patients. '
                      'Model predicts creatinine_t+1 from creatinine_t, BP_t, age_t.'
            ),
            
            'liver': OrganSystemInfo(
                name='Liver System',
                parameters=['ALT', 'AST'],
                learning_status=LearningStatus.PLACEHOLDER,
                data_source='NHANES (data quality issue - constant values)',
                sample_size=0,  # No usable temporal transitions
                temporal_coverage=None,
                notes='❌ PLACEHOLDER: ALT and AST are constant (28, 25) for all patients in NHANES. '
                      'This is a data quality issue, not biological reality. '
                      'Cannot learn temporal dynamics from constant data. '
                      'AWAITING: Framingham Heart Study + UK Biobank for real liver trajectories. '
                      'Current approach: Population baseline only, no trajectory prediction.'
            ),
            
            'immune': OrganSystemInfo(
                name='Immune System',
                parameters=['WBC'],
                learning_status=LearningStatus.PLACEHOLDER,
                data_source='NHANES (data quality issue - constant values)',
                sample_size=0,
                temporal_coverage=None,
                notes='❌ PLACEHOLDER: WBC is constant (1.0) for all patients in NHANES. '
                      'This is a data quality issue. '
                      'Cannot learn temporal dynamics from constant data. '
                      'AWAITING: UK Biobank for real immune trajectories. '
                      'Current approach: Population baseline only, no trajectory prediction.'
            ),
            
            'neural': OrganSystemInfo(
                name='Neural System',
                parameters=['cognitive_score'],
                learning_status=LearningStatus.PLACEHOLDER,
                data_source='NHANES (data quality issue - constant values)',
                sample_size=0,
                temporal_coverage=None,
                notes='❌ PLACEHOLDER: Cognitive score is constant (0.5) for all patients in NHANES. '
                      'This is a data quality issue. '
                      'Cannot learn temporal dynamics from constant data. '
                      'AWAITING: BLSA (Baltimore Longitudinal Study of Aging) + UK Biobank. '
                      'Current approach: Population baseline only, no trajectory prediction.'
            ),
            
            'lifestyle': OrganSystemInfo(
                name='Lifestyle Factors',
                parameters=['exercise_frequency', 'alcohol_consumption', 'diet_quality', 'sleep_hours'],
                learning_status=LearningStatus.PLACEHOLDER,
                data_source='NHANES (data quality issue - constant values)',
                sample_size=0,
                temporal_coverage=None,
                notes='❌ PLACEHOLDER: All lifestyle factors are constant (0, 0, 0, 7) in NHANES. '
                      'This is a data quality issue. '
                      'Cannot learn temporal dynamics from constant data. '
                      'AWAITING: Framingham + UK Biobank for real lifestyle trajectories. '
                      'Current approach: User input required, or reverse-inference from biomarkers.'
            ),
        }
    
    def get_learned_organs(self) -> List[str]:
        """Get list of organs with true temporal learning"""
        return [
            name for name, info in self.organs.items()
            if info.learning_status == LearningStatus.TEMPORAL_LEARNED
        ]
    
    def get_placeholder_organs(self) -> List[str]:
        """Get list of organs awaiting longitudinal data"""
        return [
            name for name, info in self.organs.items()
            if info.learning_status == LearningStatus.PLACEHOLDER
        ]
    
    def get_digital_twin_completeness(self) -> float:
        """Calculate what percentage is true digital twin"""
        total = len(self.organs)
        learned = len(self.get_learned_organs())
        return (learned / total) * 100
    
    def print_status_report(self):
        """Print comprehensive status report"""
        print("="*80)
        print("ORGAN SYSTEM REGISTRY - LEARNING STATUS")
        print("="*80)
        
        learned = self.get_learned_organs()
        placeholder = self.get_placeholder_organs()
        completeness = self.get_digital_twin_completeness()
        
        print(f"\nDigital Twin Completeness: {completeness:.0f}%")
        print(f"Learned Organs: {len(learned)}/{len(self.organs)}")
        print(f"Placeholder Organs: {len(placeholder)}/{len(self.organs)}")
        
        print("\n" + "="*80)
        print("LEARNED ORGANS (True Digital Twin)")
        print("="*80)
        
        for organ_name in learned:
            info = self.organs[organ_name]
            print(f"\n✅ {info.name.upper()}")
            print(f"   Parameters: {', '.join(info.parameters)}")
            print(f"   Data Source: {info.data_source}")
            print(f"   Sample Size: {info.sample_size:,} temporal transitions")
            print(f"   Coverage: {info.temporal_coverage}")
            print(f"   Notes: {info.notes}")
        
        print("\n" + "="*80)
        print("PLACEHOLDER ORGANS (Awaiting Longitudinal Data)")
        print("="*80)
        
        for organ_name in placeholder:
            info = self.organs[organ_name]
            print(f"\n❌ {info.name.upper()}")
            print(f"   Parameters: {', '.join(info.parameters)}")
            print(f"   Issue: {info.data_source}")
            print(f"   Notes: {info.notes}")
        
        print("\n" + "="*80)
        print("NEXT STEPS FOR FULL DIGITAL TWIN")
        print("="*80)
        print("\n1. Apply for Framingham Heart Study (dbGaP)")
        print("   → Provides: Liver trajectories, Lifestyle trajectories")
        print("   → Timeline: 2-3 months to data access")
        print("\n2. Apply for UK Biobank")
        print("   → Provides: Liver, Immune, Neural trajectories")
        print("   → Timeline: 1-2 months to data access")
        print("\n3. Apply for 45 and Up Study (Australia)")
        print("   → Provides: Australian population validation")
        print("   → Timeline: 1-2 months to data access")
        print("\n4. Apply for BLSA (Baltimore Longitudinal Study of Aging)")
        print("   → Provides: Neural/cognitive trajectories")
        print("   → Timeline: 2-3 months to data access")
        
        print("\n" + "="*80)
        print("EXPECTED OUTCOME AFTER DATA ACQUISITION")
        print("="*80)
        print("\nWith Framingham + UK Biobank + BLSA:")
        print("  Digital Twin Completeness: 100%")
        print("  All 7 organ systems learned from real longitudinal trajectories")
        print("  Ready for Nature Digital Medicine / npj Digital Medicine submission")
        print("\n" + "="*80)
    
    def can_predict_trajectory(self, organ: str) -> bool:
        """Check if we can predict temporal trajectory for this organ"""
        if organ not in self.organs:
            return False
        return self.organs[organ].learning_status == LearningStatus.TEMPORAL_LEARNED
    
    def get_prediction_warning(self, organ: str) -> Optional[str]:
        """Get warning message if organ cannot predict trajectories"""
        if organ not in self.organs:
            return "Unknown organ system"
        
        info = self.organs[organ]
        if info.learning_status == LearningStatus.PLACEHOLDER:
            return (f"⚠️  WARNING: {info.name} cannot predict temporal trajectories. "
                   f"Data quality issue in NHANES (constant values). "
                   f"Awaiting longitudinal cohort data for true digital twin learning.")
        
        return None


def main():
    """Print status report"""
    registry = OrganSystemRegistry()
    registry.print_status_report()
    
    print("\n\nTESTING TRAJECTORY PREDICTION CAPABILITY:")
    print("="*80)
    
    for organ in ['metabolic', 'liver', 'immune']:
        can_predict = registry.can_predict_trajectory(organ)
        warning = registry.get_prediction_warning(organ)
        
        print(f"\n{organ.upper()}:")
        print(f"  Can predict trajectory: {can_predict}")
        if warning:
            print(f"  {warning}")


if __name__ == '__main__':
    main()
