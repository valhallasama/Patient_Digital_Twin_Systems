"""
Continuous Learning System
Automated model retraining, online learning, and self-improvement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import json
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Track model performance over time"""
    model_name: str
    timestamp: datetime
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    n_samples: int
    data_drift_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'roc_auc': self.roc_auc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'n_samples': self.n_samples,
            'data_drift_score': self.data_drift_score
        }


@dataclass
class RetrainingTrigger:
    """Conditions that trigger model retraining"""
    performance_degradation_threshold: float = 0.05  # 5% drop in ROC-AUC
    data_drift_threshold: float = 0.3
    min_new_samples: int = 10000
    max_days_since_training: int = 90
    force_retrain: bool = False


class ContinuousLearningEngine:
    """
    Manages continuous learning and automated model retraining
    Monitors performance, detects drift, triggers retraining
    """
    
    def __init__(self, models_path: Path = None, data_path: Path = None):
        self.models_path = models_path or Path("models/continuous")
        self.data_path = data_path or Path("data/continuous_learning")
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Track model versions
        self.model_registry: Dict[str, List[Dict]] = {}
        
        # Performance history
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        
        # New data buffer (for incremental learning)
        self.new_data_buffer: Dict[str, pd.DataFrame] = {}
        
        # Retraining queue
        self.retraining_queue: List[str] = []
        
        # Load registry if exists
        self._load_registry()
        
        logger.info("✓ Continuous Learning Engine initialized")
        logger.info(f"  Models path: {self.models_path}")
        logger.info(f"  Data path: {self.data_path}")
    
    def register_model(self, model_name: str, model_type: str, 
                      initial_metrics: ModelPerformanceMetrics):
        """Register a model for continuous learning"""
        if model_name not in self.model_registry:
            self.model_registry[model_name] = []
        
        version = {
            'version': len(self.model_registry[model_name]) + 1,
            'model_type': model_type,
            'trained_at': datetime.now().isoformat(),
            'metrics': initial_metrics.to_dict(),
            'status': 'active'
        }
        
        self.model_registry[model_name].append(version)
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(initial_metrics)
        
        self._save_registry()
        
        logger.info(f"✓ Registered model: {model_name} v{version['version']}")
    
    def add_new_data(self, model_name: str, data: pd.DataFrame):
        """Add new data to buffer for incremental learning"""
        if model_name not in self.new_data_buffer:
            self.new_data_buffer[model_name] = data
        else:
            self.new_data_buffer[model_name] = pd.concat([
                self.new_data_buffer[model_name],
                data
            ], ignore_index=True)
        
        logger.info(f"Added {len(data)} samples to {model_name} buffer")
        logger.info(f"  Total buffered: {len(self.new_data_buffer[model_name])}")
    
    def evaluate_model_performance(self, model_name: str, model: Any,
                                  test_data: pd.DataFrame, 
                                  target_col: str) -> ModelPerformanceMetrics:
        """Evaluate current model performance"""
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            timestamp=datetime.now(),
            accuracy=accuracy_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            n_samples=len(test_data)
        )
        
        # Add to history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(metrics)
        
        logger.info(f"Model performance: {model_name}")
        logger.info(f"  ROC-AUC: {metrics.roc_auc:.3f}")
        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        
        return metrics
    
    def detect_performance_degradation(self, model_name: str, 
                                      threshold: float = 0.05) -> Tuple[bool, float]:
        """
        Detect if model performance has degraded
        Returns (degraded, drop_amount)
        """
        if model_name not in self.performance_history:
            return False, 0.0
        
        history = self.performance_history[model_name]
        if len(history) < 2:
            return False, 0.0
        
        # Compare latest to best historical performance
        latest_roc_auc = history[-1].roc_auc
        best_roc_auc = max(m.roc_auc for m in history[:-1])
        
        drop = best_roc_auc - latest_roc_auc
        degraded = drop > threshold
        
        if degraded:
            logger.warning(f"⚠ Performance degradation detected for {model_name}")
            logger.warning(f"  Best ROC-AUC: {best_roc_auc:.3f}")
            logger.warning(f"  Current ROC-AUC: {latest_roc_auc:.3f}")
            logger.warning(f"  Drop: {drop:.3f}")
        
        return degraded, drop
    
    def detect_data_drift(self, model_name: str, 
                         reference_data: pd.DataFrame,
                         new_data: pd.DataFrame) -> float:
        """
        Detect data drift using statistical tests
        Returns drift score (0-1, higher = more drift)
        """
        from scipy.stats import ks_2samp
        
        # Compare distributions of numeric features
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        drift_scores = []
        for col in numeric_cols:
            if col in new_data.columns:
                # Kolmogorov-Smirnov test
                statistic, pvalue = ks_2samp(
                    reference_data[col].dropna(),
                    new_data[col].dropna()
                )
                drift_scores.append(statistic)
        
        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        logger.info(f"Data drift score for {model_name}: {avg_drift:.3f}")
        
        return avg_drift
    
    def should_retrain(self, model_name: str, 
                      trigger: RetrainingTrigger) -> Tuple[bool, List[str]]:
        """
        Determine if model should be retrained
        Returns (should_retrain, reasons)
        """
        reasons = []
        
        # Force retrain
        if trigger.force_retrain:
            reasons.append("Force retrain requested")
            return True, reasons
        
        # Check performance degradation
        degraded, drop = self.detect_performance_degradation(
            model_name, 
            trigger.performance_degradation_threshold
        )
        if degraded:
            reasons.append(f"Performance degraded by {drop:.3f}")
        
        # Check new data availability
        if model_name in self.new_data_buffer:
            n_new = len(self.new_data_buffer[model_name])
            if n_new >= trigger.min_new_samples:
                reasons.append(f"{n_new} new samples available")
        
        # Check time since last training
        if model_name in self.model_registry and self.model_registry[model_name]:
            last_trained = datetime.fromisoformat(
                self.model_registry[model_name][-1]['trained_at']
            )
            days_since = (datetime.now() - last_trained).days
            
            if days_since >= trigger.max_days_since_training:
                reasons.append(f"{days_since} days since last training")
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            logger.info(f"✓ Retraining recommended for {model_name}")
            for reason in reasons:
                logger.info(f"  • {reason}")
        
        return should_retrain, reasons
    
    def retrain_model(self, model_name: str, model_type: str,
                     training_data: pd.DataFrame, target_col: str,
                     validation_data: pd.DataFrame) -> Tuple[Any, ModelPerformanceMetrics]:
        """
        Retrain model with new data
        Returns (new_model, metrics)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"RETRAINING MODEL: {model_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Training samples: {len(training_data)}")
        logger.info(f"Validation samples: {len(validation_data)}")
        
        # Prepare features
        feature_cols = [col for col in training_data.columns if col != target_col]
        X_train = training_data[feature_cols]
        y_train = training_data[target_col]
        
        # Train model based on type
        if model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        logger.info("Training...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model_performance(
            model_name, model, validation_data, target_col
        )
        
        # Save model
        model_path = self.models_path / f"{model_name}_v{len(self.model_registry.get(model_name, [])) + 1}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"✓ Model saved: {model_path}")
        
        # Register new version
        self.register_model(model_name, model_type, metrics)
        
        # Clear buffer
        if model_name in self.new_data_buffer:
            del self.new_data_buffer[model_name]
        
        return model, metrics
    
    def run_continuous_learning_cycle(self, trigger: RetrainingTrigger):
        """
        Run one cycle of continuous learning
        Check all models and retrain if needed
        """
        logger.info("\n" + "="*80)
        logger.info("CONTINUOUS LEARNING CYCLE")
        logger.info("="*80)
        
        retrained_models = []
        
        for model_name in self.model_registry.keys():
            should_retrain, reasons = self.should_retrain(model_name, trigger)
            
            if should_retrain:
                logger.info(f"\nRetraining {model_name}...")
                logger.info(f"Reasons: {', '.join(reasons)}")
                
                # Add to queue
                if model_name not in self.retraining_queue:
                    self.retraining_queue.append(model_name)
                    retrained_models.append(model_name)
        
        logger.info(f"\n✓ Continuous learning cycle complete")
        logger.info(f"  Models queued for retraining: {len(retrained_models)}")
        
        return retrained_models
    
    def get_model_lineage(self, model_name: str) -> List[Dict]:
        """Get full version history of a model"""
        return self.model_registry.get(model_name, [])
    
    def get_performance_trend(self, model_name: str) -> pd.DataFrame:
        """Get performance trend over time"""
        if model_name not in self.performance_history:
            return pd.DataFrame()
        
        data = [m.to_dict() for m in self.performance_history[model_name]]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def export_learning_report(self) -> Dict:
        """Export comprehensive learning report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.model_registry),
            'models': {}
        }
        
        for model_name in self.model_registry.keys():
            lineage = self.get_model_lineage(model_name)
            trend = self.get_performance_trend(model_name)
            
            report['models'][model_name] = {
                'total_versions': len(lineage),
                'current_version': lineage[-1] if lineage else None,
                'performance_trend': trend.to_dict('records') if not trend.empty else [],
                'buffered_samples': len(self.new_data_buffer.get(model_name, []))
            }
        
        return report
    
    def _save_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_path / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _load_registry(self):
        """Load model registry from disk"""
        registry_path = self.models_path / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)
            logger.info(f"✓ Loaded registry: {len(self.model_registry)} models")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONTINUOUS LEARNING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize engine
    engine = ContinuousLearningEngine()
    
    # Simulate initial model
    initial_metrics = ModelPerformanceMetrics(
        model_name='diabetes_predictor',
        timestamp=datetime.now(),
        accuracy=0.85,
        roc_auc=0.88,
        precision=0.82,
        recall=0.78,
        f1_score=0.80,
        n_samples=50000
    )
    
    engine.register_model('diabetes_predictor', 'gradient_boosting', initial_metrics)
    
    # Simulate performance degradation
    degraded_metrics = ModelPerformanceMetrics(
        model_name='diabetes_predictor',
        timestamp=datetime.now(),
        accuracy=0.78,
        roc_auc=0.81,  # Dropped by 0.07
        precision=0.75,
        recall=0.72,
        f1_score=0.73,
        n_samples=10000
    )
    
    engine.performance_history['diabetes_predictor'].append(degraded_metrics)
    
    # Check if retraining needed
    trigger = RetrainingTrigger(
        performance_degradation_threshold=0.05,
        min_new_samples=5000
    )
    
    should_retrain, reasons = engine.should_retrain('diabetes_predictor', trigger)
    
    print(f"\nShould retrain: {should_retrain}")
    print(f"Reasons: {reasons}")
    
    # Run continuous learning cycle
    retrained = engine.run_continuous_learning_cycle(trigger)
    
    print(f"\nModels queued for retraining: {retrained}")
    
    # Export report
    report = engine.export_learning_report()
    print(f"\nLearning Report:")
    print(f"  Total models: {report['total_models']}")
    for model_name, info in report['models'].items():
        print(f"  {model_name}:")
        print(f"    Versions: {info['total_versions']}")
        print(f"    Performance entries: {len(info['performance_trend'])}")
    
    print("\n" + "="*80)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*80)
