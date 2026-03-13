"""
LSTM-based Patient Trajectory Predictor
Learns from real patient data to predict disease progression
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


class PatientTrajectoryDataset(Dataset):
    """Dataset for patient lab trajectories"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PatientLSTM(nn.Module):
    """
    LSTM model for predicting patient lab trajectories
    
    Architecture:
    - Input: Sequence of lab values (glucose, HbA1c, eGFR, etc.)
    - LSTM layers: Capture temporal dependencies
    - Output: Future lab values
    """
    
    def __init__(
        self,
        input_size: int = 6,  # glucose, hba1c, egfr, creatinine, sbp, dbp
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 6,
        dropout: float = 0.2
    ):
        super(PatientLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            predictions: (batch_size, sequence_length, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Apply output layer to each timestep
        predictions = self.fc(lstm_out)  # (batch, seq, output)
        
        return predictions


class PatientPredictor:
    """
    Wrapper for training and using the LSTM predictor
    """
    
    def __init__(
        self,
        model_path: str = "models/checkpoints/patient_lstm.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.scaler_params = None
        
        print(f"🖥️  Using device: {device}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the LSTM model on patient data
        """
        print("\n🏋️  Training LSTM on patient trajectories...")
        
        # Normalize data
        self.scaler_params = self._fit_scaler(X_train)
        X_train_norm = self._normalize(X_train)
        y_train_norm = self._normalize(y_train)
        X_val_norm = self._normalize(X_val)
        y_val_norm = self._normalize(y_val)
        
        # Create datasets
        train_dataset = PatientTrajectoryDataset(X_train_norm, y_train_norm)
        val_dataset = PatientTrajectoryDataset(X_val_norm, y_val_norm)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = PatientLSTM(input_size=input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
        
        print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    
    def predict(self, patient_history: np.ndarray) -> np.ndarray:
        """
        Predict future lab values given patient history
        
        Args:
            patient_history: (sequence_length, num_features) array of past labs
        
        Returns:
            predictions: (sequence_length, num_features) array of future labs
        """
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        # Normalize input
        history_norm = self._normalize(patient_history[np.newaxis, :, :])
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(history_norm).to(self.device)
            predictions_norm = self.model(X_tensor).cpu().numpy()
        
        # Denormalize
        predictions = self._denormalize(predictions_norm[0])
        
        return predictions
    
    def _fit_scaler(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit normalization parameters"""
        # Reshape to (samples * timesteps, features)
        X_flat = X.reshape(-1, X.shape[2])
        
        mean = np.mean(X_flat, axis=0)
        std = np.std(X_flat, axis=0)
        
        return {'mean': mean, 'std': std}
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data"""
        if self.scaler_params is None:
            return X
        
        mean = self.scaler_params['mean']
        std = self.scaler_params['std']
        
        return (X - mean) / (std + 1e-8)
    
    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        """Denormalize data"""
        if self.scaler_params is None:
            return X
        
        mean = self.scaler_params['mean']
        std = self.scaler_params['std']
        
        return X * std + mean
    
    def save_model(self):
        """Save model and scaler parameters"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_params': self.scaler_params
        }, self.model_path)
    
    def load_model(self):
        """Load model and scaler parameters"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model with same architecture
        input_size = len(checkpoint['scaler_params']['mean'])
        self.model = PatientLSTM(input_size=input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_params = checkpoint['scaler_params']
        
        print(f"✓ Model loaded from {self.model_path}")


def get_patient_predictor() -> PatientPredictor:
    """Get or create patient predictor"""
    return PatientPredictor()


if __name__ == "__main__":
    # Test the predictor
    print("Testing LSTM predictor...")
    
    # Create dummy data
    n_samples = 1000
    seq_length = 30
    n_features = 6
    
    X = np.random.randn(n_samples, seq_length, n_features)
    y = np.random.randn(n_samples, seq_length, n_features)
    
    # Split train/val
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train
    predictor = get_patient_predictor()
    predictor.train(X_train, y_train, X_val, y_val, epochs=20)
    
    # Test prediction
    test_history = X_val[0]
    predictions = predictor.predict(test_history)
    
    print(f"\n✓ Prediction shape: {predictions.shape}")
    print("✅ LSTM predictor ready!")
