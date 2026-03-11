"""
Temporal ML Models - Time-series and Survival Analysis
For longitudinal health prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurvivalAnalysisModel:
    """
    Survival analysis for time-to-event prediction
    Uses Cox Proportional Hazards and Kaplan-Meier
    """
    
    def __init__(self):
        self.models = {}
        self.baseline_hazards = {}
        
        # Try to import survival analysis libraries
        try:
            from lifelines import CoxPHFitter, KaplanMeierFitter
            from lifelines.utils import concordance_index
            self.CoxPHFitter = CoxPHFitter
            self.KaplanMeierFitter = KaplanMeierFitter
            self.concordance_index = concordance_index
            self.available = True
            logger.info("✓ Survival analysis libraries available")
        except ImportError:
            logger.warning("lifelines not installed. Survival analysis unavailable.")
            self.available = False
    
    def train_cox_model(self, data: pd.DataFrame, duration_col: str, 
                       event_col: str, feature_cols: List[str],
                       model_name: str = 'default') -> Dict:
        """
        Train Cox Proportional Hazards model
        
        Args:
            data: DataFrame with patient data
            duration_col: Time to event or censoring
            event_col: Event indicator (1=event, 0=censored)
            feature_cols: List of feature column names
            model_name: Name for this model
        """
        
        if not self.available:
            logger.error("Survival analysis not available")
            return {'error': 'lifelines not installed'}
        
        logger.info(f"\nTraining Cox PH model: {model_name}")
        logger.info(f"Dataset: {len(data)} patients")
        logger.info(f"Events: {data[event_col].sum()} ({data[event_col].mean():.1%})")
        
        # Prepare data
        model_data = data[[duration_col, event_col] + feature_cols].copy()
        model_data = model_data.dropna()
        
        # Train model
        cph = self.CoxPHFitter()
        cph.fit(model_data, duration_col=duration_col, event_col=event_col)
        
        # Store model
        self.models[model_name] = cph
        
        # Calculate concordance index (C-index)
        predictions = cph.predict_partial_hazard(model_data[feature_cols])
        c_index = self.concordance_index(
            model_data[duration_col],
            -predictions,  # Negative because higher hazard = worse outcome
            model_data[event_col]
        )
        
        logger.info(f"✓ C-index: {c_index:.3f}")
        
        # Get hazard ratios
        hazard_ratios = np.exp(cph.params_)
        
        return {
            'model': cph,
            'c_index': c_index,
            'hazard_ratios': hazard_ratios.to_dict(),
            'summary': cph.summary,
            'n_patients': len(model_data),
            'n_events': model_data[event_col].sum()
        }
    
    def predict_survival_function(self, patient_features: pd.DataFrame,
                                  model_name: str = 'default',
                                  times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Predict survival function for patient(s)
        Returns probability of survival at each time point
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        cph = self.models[model_name]
        survival_func = cph.predict_survival_function(patient_features, times=times)
        
        return survival_func
    
    def predict_median_survival(self, patient_features: pd.DataFrame,
                               model_name: str = 'default') -> pd.Series:
        """Predict median survival time for patient(s)"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        cph = self.models[model_name]
        median_survival = cph.predict_median(patient_features)
        
        return median_survival
    
    def calculate_risk_score(self, patient_features: pd.DataFrame,
                            model_name: str = 'default') -> pd.Series:
        """Calculate risk score (partial hazard) for patient(s)"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        cph = self.models[model_name]
        risk_scores = cph.predict_partial_hazard(patient_features)
        
        return risk_scores
    
    def fit_kaplan_meier(self, durations: np.ndarray, events: np.ndarray,
                        label: str = 'KM') -> Dict:
        """
        Fit Kaplan-Meier survival curve
        Non-parametric survival estimation
        """
        
        if not self.available:
            return {'error': 'lifelines not installed'}
        
        kmf = self.KaplanMeierFitter()
        kmf.fit(durations, events, label=label)
        
        return {
            'model': kmf,
            'median_survival': kmf.median_survival_time_,
            'survival_function': kmf.survival_function_,
            'confidence_interval': kmf.confidence_interval_
        }


class LSTMHealthPredictor:
    """
    LSTM model for time-series health prediction
    Predicts future health states from historical data
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.model = None
        
        # Try to import PyTorch
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self.available = True
            logger.info("✓ PyTorch available for LSTM models")
        except ImportError:
            logger.warning("PyTorch not installed. LSTM models unavailable.")
            self.available = False
    
    def build_model(self):
        """Build LSTM architecture"""
        
        if not self.available:
            logger.error("PyTorch not available")
            return None
        
        class LSTMModel(self.nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # LSTM layers
                self.lstm = self.nn.LSTM(
                    input_dim, 
                    hidden_dim, 
                    num_layers, 
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0
                )
                
                # Fully connected output layer
                self.fc = self.nn.Linear(hidden_dim, output_dim)
                self.sigmoid = self.nn.Sigmoid()
            
            def forward(self, x):
                # x shape: (batch, seq_len, input_dim)
                lstm_out, (h_n, c_n) = self.lstm(x)
                
                # Use last time step output
                last_output = lstm_out[:, -1, :]
                
                # Fully connected layer
                output = self.fc(last_output)
                output = self.sigmoid(output)
                
                return output
        
        self.model = LSTMModel(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            self.output_dim
        )
        
        logger.info(f"✓ LSTM model built: {self.input_dim}→{self.hidden_dim}→{self.output_dim}")
        return self.model
    
    def prepare_sequences(self, data: pd.DataFrame, 
                         sequence_length: int = 12,
                         target_col: str = 'target') -> Tuple:
        """
        Prepare time-series sequences for LSTM
        
        Args:
            data: DataFrame with time-series data
            sequence_length: Number of time steps in each sequence
            target_col: Target variable column
        
        Returns:
            X, y arrays for training
        """
        
        if not self.available:
            return None, None
        
        # Group by patient
        patient_ids = data['patient_id'].unique()
        
        sequences = []
        targets = []
        
        for patient_id in patient_ids:
            patient_data = data[data['patient_id'] == patient_id].sort_values('timestamp')
            
            if len(patient_data) < sequence_length + 1:
                continue
            
            # Create sequences
            for i in range(len(patient_data) - sequence_length):
                seq = patient_data.iloc[i:i+sequence_length]
                target = patient_data.iloc[i+sequence_length][target_col]
                
                # Extract features (exclude patient_id, timestamp, target)
                feature_cols = [c for c in seq.columns 
                               if c not in ['patient_id', 'timestamp', target_col]]
                
                sequences.append(seq[feature_cols].values)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"Prepared {len(X)} sequences of length {sequence_length}")
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32,
             learning_rate: float = 0.001) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        
        if not self.available or self.model is None:
            logger.error("Model not available or not built")
            return {'error': 'Model not available'}
        
        # Convert to PyTorch tensors
        X_train_t = self.torch.FloatTensor(X_train)
        y_train_t = self.torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = self.torch.FloatTensor(X_val)
        y_val_t = self.torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = self.torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = self.torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Loss and optimizer
        criterion = self.nn.BCELoss()
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"\nTraining LSTM for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with self.torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss.item():.4f}")
        
        logger.info("✓ Training complete")
        
        return {
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on sequences"""
        
        if not self.available or self.model is None:
            return None
        
        self.model.eval()
        with self.torch.no_grad():
            X_t = self.torch.FloatTensor(X)
            predictions = self.model(X_t)
        
        return predictions.numpy()


class TemporalRiskPredictor:
    """
    Combines survival analysis and LSTM for comprehensive temporal prediction
    """
    
    def __init__(self):
        self.survival_model = SurvivalAnalysisModel()
        self.lstm_models = {}
    
    def train_disease_onset_model(self, patient_timelines: pd.DataFrame,
                                  disease: str = 'diabetes') -> Dict:
        """
        Train model to predict time to disease onset
        
        Args:
            patient_timelines: DataFrame with columns:
                - patient_id
                - time_to_event (years until disease or censoring)
                - event (1 if disease occurred, 0 if censored)
                - features (age, bmi, etc.)
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Temporal Model: {disease.upper()} Onset Prediction")
        logger.info(f"{'='*80}")
        
        # Feature columns
        feature_cols = [col for col in patient_timelines.columns 
                       if col not in ['patient_id', 'time_to_event', 'event']]
        
        # Train Cox model
        cox_results = self.survival_model.train_cox_model(
            data=patient_timelines,
            duration_col='time_to_event',
            event_col='event',
            feature_cols=feature_cols,
            model_name=f'{disease}_onset'
        )
        
        return {
            'disease': disease,
            'model_type': 'cox_proportional_hazards',
            'c_index': cox_results.get('c_index'),
            'n_patients': cox_results.get('n_patients'),
            'n_events': cox_results.get('n_events'),
            'hazard_ratios': cox_results.get('hazard_ratios')
        }
    
    def predict_disease_risk_over_time(self, patient_features: pd.DataFrame,
                                       disease: str = 'diabetes',
                                       years: int = 10) -> pd.DataFrame:
        """
        Predict disease risk trajectory over time
        
        Returns:
            DataFrame with survival probabilities at each year
        """
        
        model_name = f'{disease}_onset'
        times = np.arange(0, years + 1)
        
        survival_func = self.survival_model.predict_survival_function(
            patient_features,
            model_name=model_name,
            times=times
        )
        
        # Convert survival to risk
        risk_over_time = 1 - survival_func
        
        return risk_over_time


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEMPORAL ML MODELS DEMONSTRATION")
    print("="*80)
    
    # Example: Survival Analysis
    print("\n1. SURVIVAL ANALYSIS - Cox Proportional Hazards")
    print("-" * 80)
    
    # Simulate patient data
    np.random.seed(42)
    n_patients = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(55, 10, n_patients),
        'bmi': np.random.normal(28, 5, n_patients),
        'systolic_bp': np.random.normal(135, 15, n_patients),
        'glucose': np.random.normal(5.8, 1.2, n_patients),
        'smoking': np.random.binomial(1, 0.3, n_patients)
    })
    
    # Simulate time to diabetes onset
    baseline_hazard = 0.05
    hazard = baseline_hazard * np.exp(
        0.05 * data['age'] + 
        0.1 * data['bmi'] + 
        0.02 * data['systolic_bp'] +
        0.5 * data['smoking']
    )
    
    data['time_to_event'] = np.random.exponential(1/hazard)
    data['time_to_event'] = np.clip(data['time_to_event'], 0, 10)
    data['event'] = (data['time_to_event'] < 10).astype(int)
    
    # Train survival model
    survival_model = SurvivalAnalysisModel()
    
    if survival_model.available:
        results = survival_model.train_cox_model(
            data=data,
            duration_col='time_to_event',
            event_col='event',
            feature_cols=['age', 'bmi', 'systolic_bp', 'glucose', 'smoking'],
            model_name='diabetes_onset'
        )
        
        print(f"\nModel Performance:")
        print(f"  C-index: {results['c_index']:.3f}")
        print(f"  Patients: {results['n_patients']}")
        print(f"  Events: {results['n_events']}")
        
        print(f"\nHazard Ratios:")
        for feature, hr in results['hazard_ratios'].items():
            print(f"  {feature}: {hr:.3f}")
        
        # Predict for new patient
        new_patient = pd.DataFrame({
            'age': [60],
            'bmi': [32],
            'systolic_bp': [145],
            'glucose': [6.5],
            'smoking': [1]
        })
        
        median_time = survival_model.predict_median_survival(
            new_patient, 
            model_name='diabetes_onset'
        )
        print(f"\nPrediction for high-risk patient:")
        print(f"  Median time to diabetes: {median_time.values[0]:.1f} years")
    
    print("\n" + "="*80)
    print("✓ Temporal models demonstration complete")
    print("="*80)
