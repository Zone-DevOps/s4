"""
Transformation functions for System 4 triadic processing
"""

import numpy as np
from typing import Dict, Tuple, Any

class TransformationEngine:
    """
    Manages transformation rules and operations for the triadic system
    """
    
    def __init__(self):
        self.transformation_count = 0
        self.transformation_history = []
        
    def apply_past_transformation(self, state: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """
        Transform Past_Actual state based on performance measurement
        """
        # Consolidate with decay
        transformed = state * 0.95
        
        # Incorporate alignment feedback
        alignment = metrics.get("alignment", 0.0)
        transformed += 0.05 * alignment
        
        # Apply normalization
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed /= norm
            
        return transformed
    
    def apply_present_transformation(self, state: np.ndarray, interference: np.ndarray) -> np.ndarray:
        """
        Transform Present_Real state through relevance realization
        """
        # Blend current state with interference pattern
        transformed = 0.7 * state + 0.3 * interference
        
        # Apply sigmoid activation for bounded output
        transformed = np.tanh(transformed)
        
        return transformed
    
    def apply_future_transformation(self, state: np.ndarray, exploration_rate: float = 0.01) -> np.ndarray:
        """
        Transform Future_Virtual state through exploratory projection
        """
        # Add exploration noise
        noise = np.random.randn(len(state)) * exploration_rate
        transformed = state + noise
        
        # Apply bounded activation
        transformed = np.tanh(transformed)
        
        return transformed
    
    def compute_triadic_coupling(self, past: np.ndarray, present: np.ndarray, future: np.ndarray) -> Dict[str, Any]:
        """
        Compute coupling coefficients between the three temporal streams
        """
        # Calculate pairwise correlations
        past_present = np.corrcoef(past, present)[0, 1]
        present_future = np.corrcoef(present, future)[0, 1]
        past_future = np.corrcoef(past, future)[0, 1]
        
        # Calculate triadic coherence
        coherence = np.abs(past_present * present_future * past_future)
        
        # Calculate phase relationships
        phase_diff_pp = np.angle(np.vdot(past, present))
        phase_diff_pf = np.angle(np.vdot(present, future))
        
        return {
            "past_present_correlation": float(past_present),
            "present_future_correlation": float(present_future),
            "past_future_correlation": float(past_future),
            "triadic_coherence": float(coherence),
            "phase_past_present": float(phase_diff_pp),
            "phase_present_future": float(phase_diff_pf)
        }
    
    def generate_interference_weights(self, step: int, total_steps: int) -> Tuple[float, float, float]:
        """
        Generate phase-dependent weights for interference pattern
        """
        phase = 2 * np.pi * step / total_steps
        
        # Past weight: cosine modulation
        past_weight = np.cos(phase)
        
        # Present weight: always full
        present_weight = 1.0
        
        # Future weight: sine modulation
        future_weight = np.sin(phase)
        
        return past_weight, present_weight, future_weight
    
    def log_transformation(self, set_name: str, before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        """
        Log transformation details for analysis
        """
        delta = after - before
        magnitude_change = np.linalg.norm(after) - np.linalg.norm(before)
        direction_change = 1.0 - np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-10)
        
        log_entry = {
            "transformation_id": self.transformation_count,
            "set_name": set_name,
            "delta_norm": float(np.linalg.norm(delta)),
            "magnitude_change": float(magnitude_change),
            "direction_change": float(direction_change),
            "max_component_change": float(np.max(np.abs(delta))),
            "mean_component_change": float(np.mean(np.abs(delta)))
        }
        
        self.transformation_count += 1
        self.transformation_history.append(log_entry)
        
        return log_entry