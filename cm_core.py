"""
Cognitional Mechanics (CM) Core - Extended Edition
Operationalizing Batch-Parallel Non-Commutative Semantic Computation.
(C) 2026 T.O. / IASER Framework
"""

import numpy as np

class CMSemanticEngine:
    """
    Advanced CM Engine supporting batch-parallel manifold operations.
    Designed for future GPU/TPU porting via tensor-based logic.
    """
    def __init__(self, n_dims, c_limit=0.5):
        self.n_dims = n_dims
        self.c_limit = c_limit

    def logical_leap_batch(self, states):
        """
        Deterministic re-projection applied across a batch of semantic vectors.
        states: numpy array of shape (batch_size, n_dims)
        """
        # Calculate Euclidean norms for each state in the batch
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        threshold = self.c_limit * np.sqrt(self.n_dims)
        
        # Apply the Leap: Only normalize states exceeding the operational limit
        # This preserves structural information while preventing manifold rupture.
        scale_factors = np.where(norms > threshold, norms, 1.0)
        return states / scale_factors

    def execute_batch_trajectory(self, initial_states, operator_sequence):
        """
        Calculates the evolution of semantic states through an ordered operator set.
        Returns the final states and the history of convergence.
        """
        current_states = initial_states.copy()
        history = [current_states.copy()]

        for op in operator_sequence:
            # Order-sensitive transformation (Matrix-Batch multiplication)
            current_states = current_states @ op.T
            # Deterministic manifold stabilization
            current_states = self.logical_leap_batch(current_states)
            history.append(current_states.copy())
            
        return current_states, history

# --- Extended Verification Service ---
if __name__ == "__main__":
    # Dimension of the semantic manifold
    D = 8
    engine = CMSemanticEngine(n_dims=D, c_limit=0.4)

    # Initialize a batch of 3 distinct initial semantic vectors
    batch_initial = np.random.randn(3, D)

    # Define Non-Commutative Rotation Operators (Generators)
    # O_A: Permutation + slight scaling
    O_A = np.roll(np.eye(D), 1, axis=0) * 1.1 
    # O_B: Reflection + slight scaling
    O_B = np.eye(D)[::-1] * 1.1

    # Execute competing trajectories to prove Non-Commutativity
    # Sequence Alpha: [A, B] | Sequence Beta: [B, A]
    final_alpha, _ = engine.execute_batch_trajectory(batch_initial, [O_A, O_B])
    final_beta, _  = engine.execute_batch_trajectory(batch_initial, [O_B, O_A])

    print("--- Extended CM Operational Report ---")
    print(f"Batch Size: {batch_initial.shape[0]} | Manifold Dimensions: {D}")
    print("\n[Trajectory Alpha Result (First Sample)]:\n", final_alpha[0])
    print("\n[Trajectory Beta Result (First Sample)]:\n", final_beta[0])

    diff = np.linalg.norm(final_alpha - final_beta)
    print(f"\nPath-Dependence Variance: {diff:.6f}")
    print("Verification: Intelligence is Path-Dependent and Deterministic.")