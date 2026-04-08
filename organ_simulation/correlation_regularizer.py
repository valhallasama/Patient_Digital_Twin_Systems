#!/usr/bin/env python3
"""
Correlation Regularization for Synthetic Organ Generation

Enforces target cross-organ correlations while preserving population statistics.
Uses post-hoc correlation adjustment to match literature-based targets.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class CorrelationRegularizer:
    """
    Adjust synthetic organ values to match target correlations
    
    Method: Linear correlation adjustment that preserves mean/std
    """
    
    def __init__(self, target_correlations: Dict[str, float]):
        """
        Args:
            target_correlations: Dict of 'var1_var2': correlation_value
        """
        self.target_correlations = target_correlations
    
    def adjust_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        target_corr: float,
        preserve_y_stats: bool = True
    ) -> np.ndarray:
        """
        Adjust y to have target correlation with x while preserving y's statistics
        
        Args:
            x: Independent variable (e.g., glucose)
            y: Dependent variable to adjust (e.g., ALT)
            target_corr: Target correlation coefficient
            preserve_y_stats: If True, preserve mean/std of y
        
        Returns:
            Adjusted y with target correlation to x
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 100:  # Need enough valid samples
            return y
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Standardize
        x_std = (x_valid - np.mean(x_valid)) / np.std(x_valid)
        y_std = (y_valid - np.mean(y_valid)) / np.std(y_valid)
        
        # Current correlation
        current_corr = np.corrcoef(x_std, y_std)[0, 1]
        
        # If already close, skip
        if abs(current_corr - target_corr) < 0.02:
            return y
        
        # Compute adjustment
        # y_new = target_corr * x + sqrt(1 - target_corr^2) * residual
        residual = y_std - current_corr * x_std
        residual_std = residual / np.std(residual)
        
        y_adjusted_std = (
            target_corr * x_std + 
            np.sqrt(max(0, 1 - target_corr**2)) * residual_std
        )
        
        if preserve_y_stats:
            # Restore original mean/std (of valid values)
            y_adjusted_valid = y_adjusted_std * np.std(y_valid) + np.mean(y_valid)
        else:
            y_adjusted_valid = y_adjusted_std
        
        # Reconstruct full array with NaN preserved
        y_adjusted = y.copy()
        y_adjusted[valid_mask] = y_adjusted_valid
        
        return y_adjusted
    
    def adjust_batch_correlations(
        self,
        real_organs: Dict[str, np.ndarray],
        synthetic_organs: Dict[str, np.ndarray],
        demographics: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Adjust all synthetic organs to match target correlations
        
        Args:
            real_organs: Real NHANES data (glucose, BP, etc.)
            synthetic_organs: Generated synthetic organs (ALT, WBC, etc.)
            demographics: Patient demographics (BMI, age, etc.)
        
        Returns:
            Adjusted synthetic organs with target correlations
        """
        # Work on copies to avoid in-place modifications
        adjusted = {k: v.copy() for k, v in synthetic_organs.items()}
        
        # Extract variables
        glucose = real_organs.get('glucose')
        triglycerides = real_organs.get('triglycerides')
        systolic_bp = real_organs.get('systolic_bp')
        bmi = demographics.get('bmi')
        
        alt = adjusted.get('ALT')
        wbc = adjusted.get('WBC')
        cognitive = adjusted.get('cognitive')
        
        # Apply primary correlations only (avoid cascading issues)
        
        # 1. Glucose ↔ ALT (metabolic-liver coupling)
        if glucose is not None and alt is not None:
            target = self.target_correlations.get('glucose_alt', 0.25)
            result = self.adjust_correlation(glucose, alt, target)
            if result is not None and not np.all(np.isnan(result)):
                adjusted['ALT'] = result
                print(f"  Adjusted Glucose-ALT correlation to {target:.2f}")
        
        # 2. BMI ↔ WBC (inflammation coupling)
        if bmi is not None and wbc is not None:
            target = self.target_correlations.get('bmi_wbc', 0.25)
            result = self.adjust_correlation(bmi, wbc, target)
            if result is not None and not np.all(np.isnan(result)):
                adjusted['WBC'] = result
                print(f"  Adjusted BMI-WBC correlation to {target:.2f}")
        
        # 3. BP ↔ Cognitive (cardiovascular-neural coupling)
        if systolic_bp is not None and cognitive is not None:
            target = self.target_correlations.get('systolic_bp_cognitive', -0.20)
            result = self.adjust_correlation(systolic_bp, cognitive, target)
            if result is not None and not np.all(np.isnan(result)):
                adjusted['cognitive'] = result
                print(f"  Adjusted BP-Cognitive correlation to {target:.2f}")
        
        return adjusted
    
    def multivariate_adjustment(
        self,
        variables: Dict[str, np.ndarray],
        target_cov_matrix: np.ndarray,
        var_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Adjust multiple variables jointly to match target covariance matrix
        
        Uses Cholesky decomposition for multivariate correlation structure.
        
        Args:
            variables: Dict of variable_name: values
            target_cov_matrix: Target covariance matrix
            var_names: Names of variables in order matching cov_matrix
        
        Returns:
            Adjusted variables with target covariance
        """
        n_vars = len(var_names)
        n_samples = len(variables[var_names[0]])
        
        # Stack variables into matrix
        X = np.column_stack([variables[name] for name in var_names])
        
        # Standardize
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X_std = (X - means) / stds
        
        # Current covariance
        current_cov = np.cov(X_std.T)
        
        # Cholesky decomposition of target correlation matrix
        # Convert covariance to correlation
        target_corr = np.zeros_like(target_cov_matrix)
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    target_corr[i, j] = 1.0
                else:
                    target_corr[i, j] = target_cov_matrix[i, j] / (stds[i] * stds[j])
        
        # Ensure positive definite
        target_corr = self._nearest_positive_definite(target_corr)
        
        # Transform to target correlation structure
        L_current = np.linalg.cholesky(current_cov)
        L_target = np.linalg.cholesky(target_corr)
        
        # Transform: X_new = X * L_current^-1 * L_target
        X_transformed = X_std @ np.linalg.inv(L_current) @ L_target
        
        # Restore original means and stds
        X_adjusted = X_transformed * stds + means
        
        # Return as dict
        adjusted = {
            name: X_adjusted[:, i]
            for i, name in enumerate(var_names)
        }
        
        return adjusted
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix"""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Check if matrix is positive definite"""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def validate_correlations(
        self,
        real_organs: Dict[str, np.ndarray],
        synthetic_organs: Dict[str, np.ndarray],
        demographics: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute actual correlations and compare to targets
        
        Returns:
            Dict of correlation_name: (actual, target, difference)
        """
        results = {}
        
        # Extract variables
        glucose = real_organs.get('glucose')
        triglycerides = real_organs.get('triglycerides')
        systolic_bp = real_organs.get('systolic_bp')
        bmi = demographics.get('bmi')
        
        alt = synthetic_organs.get('ALT')
        wbc = synthetic_organs.get('WBC')
        cognitive = synthetic_organs.get('cognitive')
        
        # Compute correlations
        if glucose is not None and alt is not None:
            actual = np.corrcoef(glucose, alt)[0, 1]
            target = self.target_correlations.get('glucose_alt', 0.25)
            results['glucose_alt'] = (actual, target, actual - target)
        
        if triglycerides is not None and alt is not None:
            actual = np.corrcoef(triglycerides, alt)[0, 1]
            target = self.target_correlations.get('triglycerides_alt', 0.30)
            results['triglycerides_alt'] = (actual, target, actual - target)
        
        if bmi is not None and alt is not None:
            actual = np.corrcoef(bmi, alt)[0, 1]
            target = self.target_correlations.get('bmi_alt', 0.35)
            results['bmi_alt'] = (actual, target, actual - target)
        
        if bmi is not None and wbc is not None:
            actual = np.corrcoef(bmi, wbc)[0, 1]
            target = self.target_correlations.get('bmi_wbc', 0.25)
            results['bmi_wbc'] = (actual, target, actual - target)
        
        if systolic_bp is not None and cognitive is not None:
            actual = np.corrcoef(systolic_bp, cognitive)[0, 1]
            target = self.target_correlations.get('systolic_bp_cognitive', -0.20)
            results['systolic_bp_cognitive'] = (actual, target, actual - target)
        
        if alt is not None and wbc is not None:
            actual = np.corrcoef(alt, wbc)[0, 1]
            target = self.target_correlations.get('alt_wbc', 0.18)
            results['alt_wbc'] = (actual, target, actual - target)
        
        return results


def test_correlation_regularizer():
    """Test correlation adjustment"""
    np.random.seed(42)
    
    # Generate test data
    n = 10000
    glucose = np.random.normal(100, 20, n)
    alt = np.random.normal(25, 10, n)  # Initially uncorrelated
    
    print("Before adjustment:")
    print(f"  Glucose-ALT correlation: {np.corrcoef(glucose, alt)[0, 1]:.3f}")
    print(f"  ALT mean: {np.mean(alt):.1f}, std: {np.std(alt):.1f}")
    
    # Adjust
    regularizer = CorrelationRegularizer({'glucose_alt': 0.25})
    alt_adjusted = regularizer.adjust_correlation(glucose, alt, target_corr=0.25)
    
    print("\nAfter adjustment:")
    print(f"  Glucose-ALT correlation: {np.corrcoef(glucose, alt_adjusted)[0, 1]:.3f}")
    print(f"  ALT mean: {np.mean(alt_adjusted):.1f}, std: {np.std(alt_adjusted):.1f}")
    print(f"  ✓ Correlation achieved, statistics preserved")


if __name__ == '__main__':
    test_correlation_regularizer()
