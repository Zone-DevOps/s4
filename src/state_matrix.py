"""
System 4 Triadic Transformation State Matrix
Based on A000081 architecture with 3 concurrent sets
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

class TriadicStateMatrix:
    """
    Represents System 4 with 9 terms forming 3 concurrent sets.
    Each set processes 4 steps simultaneously but with different temporal focus:
    - Set 1: Conditioned Past (Actual performance)
    - Set 2: Immanent Present (Real commitment) 
    - Set 3: Predicted Future (Virtual potential)
    """
    
    def __init__(self, scenario_name: str = "default"):
        self.scenario_name = scenario_name
        self.total_steps = 12  # 3 cycles of 4 steps
        self.sets = 3
        self.steps_per_set = 4
        
        # Initialize state matrix (3 sets × 4 steps × n dimensions)
        # Using 9 dimensions to represent the 9 terms of System 4
        self.state_matrix = np.zeros((3, 4, 9), dtype=np.float64)
        
        # Track current position in each concurrent set
        self.set_positions = [0, 0, 0]  # Current step for each set
        self.global_step = 0
        
        # Phase offsets for interleaving (in steps)
        self.phase_offsets = [0, 1, 2]  # Set 2 leads by 1, Set 3 leads by 2
        
        # Temporal labels
        self.set_names = ["Past_Actual", "Present_Real", "Future_Virtual"]
        
        # Performance metrics
        self.performance_history = []
        self.accuracy_metrics = {"precision": 0.0, "recall": 0.0, "alignment": 0.0}
        
    def initialize_from_scenario(self, scenario_path: Path) -> None:
        """Load initial conditions from scenario file"""
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        
        # Load initial state matrix
        if "initial_state" in scenario:
            self.state_matrix = np.array(scenario["initial_state"])
        
        # Load phase relationships
        if "phase_offsets" in scenario:
            self.phase_offsets = scenario["phase_offsets"]
        
        # Set metadata
        self.scenario_metadata = scenario.get("metadata", {})
        self.transformation_rules = scenario.get("transformation_rules", {})
        
    def get_concurrent_states(self) -> Dict[str, np.ndarray]:
        """Get current state of all three concurrent sets"""
        states = {}
        for i, name in enumerate(self.set_names):
            position = (self.set_positions[i] + self.phase_offsets[i]) % self.steps_per_set
            states[name] = self.state_matrix[i, position, :]
        return states
    
    def compute_interference_pattern(self) -> np.ndarray:
        """
        Compute the interference pattern where all three temporal streams meet.
        This is where the Arena emerges from the triadic structure.
        """
        states = self.get_concurrent_states()
        
        # Simple interference: superposition with phase-dependent weighting
        weights = np.array([
            np.cos(2 * np.pi * self.set_positions[0] / self.steps_per_set),  # Past
            1.0,  # Present (always full weight)
            np.sin(2 * np.pi * self.set_positions[2] / self.steps_per_set)   # Future
        ])
        
        interference = np.zeros(9)
        for i, (name, state) in enumerate(states.items()):
            interference += weights[i] * state
        
        # Normalize to maintain coherence
        norm = np.linalg.norm(interference)
        if norm > 0:
            interference /= norm
            
        return interference
    
    def measure_performance(self, ground_truth: np.ndarray = None) -> Dict[str, float]:
        """
        Measure alignment of internal model with external ground truth.
        This is the Past_Actual's primary function.
        """
        if ground_truth is None:
            # Use interference pattern as proxy for ground truth
            ground_truth = self.compute_interference_pattern()
        
        past_state = self.state_matrix[0, self.set_positions[0], :]
        
        # Calculate performance metrics
        alignment = np.dot(past_state, ground_truth) / (np.linalg.norm(past_state) * np.linalg.norm(ground_truth) + 1e-10)
        precision = 1.0 - np.std(past_state - ground_truth)
        accuracy = 1.0 - np.mean(np.abs(past_state - ground_truth))
        
        metrics = {
            "alignment": float(alignment),
            "precision": float(precision),
            "accuracy": float(accuracy),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.performance_history.append(metrics)
        self.accuracy_metrics = metrics
        
        return metrics
    
    def project_virtual_potential(self) -> np.ndarray:
        """
        Project potential future states based on capacity and context.
        This is the Future_Virtual's primary function.
        """
        future_state = self.state_matrix[2, self.set_positions[2], :]
        present_state = self.state_matrix[1, self.set_positions[1], :]
        
        # Simple projection: extrapolate from present trajectory
        velocity = future_state - present_state
        potential = future_state + velocity * np.exp(-self.global_step / self.total_steps)
        
        return potential
    
    def realize_commitment(self) -> Tuple[np.ndarray, float]:
        """
        Bind salience and affordance to relevance in the present.
        This is the Present_Real's primary function.
        Returns: (action_vector, relevance_score)
        """
        present_state = self.state_matrix[1, self.set_positions[1], :]
        interference = self.compute_interference_pattern()
        
        # Salience: what matters (from interference pattern)
        salience = np.abs(interference)
        
        # Affordance: what's actionable (from present state magnitude)
        affordance = np.abs(present_state)
        
        # Relevance: binding of salience and affordance
        relevance = salience * affordance
        relevance_score = np.sum(relevance)
        
        # Commitment: action vector based on relevance-weighted present state
        if relevance_score > 0:
            action = present_state * relevance / relevance_score
        else:
            action = present_state
            
        return action, float(relevance_score)
    
    def step(self) -> Dict[str, Any]:
        """
        Advance all three concurrent sets by one step.
        This creates the interleaved temporal processing.
        """
        # Convert numpy arrays to lists for JSON serialization
        before_states = {}
        for name, state in self.get_concurrent_states().items():
            before_states[name] = state.tolist()
        
        step_log = {
            "global_step": self.global_step,
            "timestamp": datetime.utcnow().isoformat(),
            "before_state": before_states,
            "set_positions": self.set_positions.copy()
        }
        
        # Apply transformations to each set
        for set_idx in range(self.sets):
            self._transform_set(set_idx)
            self.set_positions[set_idx] = (self.set_positions[set_idx] + 1) % self.steps_per_set
        
        # Measure performance after step
        performance = self.measure_performance()
        
        # Project future potential
        potential = self.project_virtual_potential()
        
        # Realize present commitment
        action, relevance = self.realize_commitment()
        
        # Convert numpy arrays to lists for JSON serialization
        after_states = {}
        for name, state in self.get_concurrent_states().items():
            after_states[name] = state.tolist()
        
        step_log.update({
            "after_state": after_states,
            "interference": self.compute_interference_pattern().tolist(),
            "performance": performance,
            "potential": potential.tolist(),
            "action": action.tolist(),
            "relevance_score": relevance,
            "cycle": self.global_step // self.steps_per_set,
            "phase": self.global_step % self.steps_per_set
        })
        
        self.global_step += 1
        
        return step_log
    
    def _transform_set(self, set_idx: int) -> None:
        """Apply transformation rules to a specific set"""
        position = self.set_positions[set_idx]
        current_state = self.state_matrix[set_idx, position, :]
        
        # Get transformation based on set type and position
        if set_idx == 0:  # Past_Actual
            # Consolidate and measure
            current_state *= 0.95  # Slight decay
            current_state += 0.05 * self.accuracy_metrics.get("alignment", 0)
        elif set_idx == 1:  # Present_Real
            # Active transformation
            interference = self.compute_interference_pattern()
            current_state = 0.7 * current_state + 0.3 * interference
        else:  # Future_Virtual
            # Exploratory projection
            current_state += np.random.randn(9) * 0.01  # Small random walk
            current_state = np.tanh(current_state)  # Bounded activation
        
        self.state_matrix[set_idx, position, :] = current_state
    
    def save_state(self, filepath: Path) -> None:
        """Save current state matrix and metadata"""
        state_data = {
            "scenario_name": self.scenario_name,
            "global_step": self.global_step,
            "state_matrix": self.state_matrix.tolist(),
            "set_positions": self.set_positions,
            "phase_offsets": self.phase_offsets,
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "accuracy_metrics": self.accuracy_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: Path) -> None:
        """Load state matrix from file"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.scenario_name = state_data["scenario_name"]
        self.global_step = state_data["global_step"]
        self.state_matrix = np.array(state_data["state_matrix"])
        self.set_positions = state_data["set_positions"]
        self.phase_offsets = state_data["phase_offsets"]
        self.performance_history = state_data.get("performance_history", [])
        self.accuracy_metrics = state_data.get("accuracy_metrics", {})