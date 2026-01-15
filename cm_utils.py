"""
CM Utilities: High-Dimensional Operator Generators
Enables the creation of structured non-commutative operator sets.
"""

import numpy as np

def generate_rotation_operator(n_dims, axis_i, axis_j, theta):
    """
    Generates a high-dimensional rotation matrix acting on a semantic plane.
    This creates an order-sensitive phase shift in the manifold.
    """
    op = np.eye(n_dims)
    c, s = np.cos(theta), np.sin(theta)
    op[axis_i, axis_i] = c
    op[axis_i, axis_j] = -s
    op[axis_j, axis_i] = s
    op[axis_j, axis_j] = c
    return op

def generate_semantic_shear(n_dims, i, j, factor):
    """
    Generates a shear transformation to represent 'contextual drift'.
    Crucially non-commutative when paired with rotations.
    """
    op = np.eye(n_dims)
    op[i, j] = factor
    return op

# Example: Building a structured operator sequence
if __name__ == "__main__":
    D = 16
    # Create two fundamental operators
    # Rotation in (0,1) plane and Shear in (1,2) plane
    R = generate_rotation_operator(D, 0, 1, np.pi/4)
    S = generate_semantic_shear(D, 1, 2, 0.5)
    
    # Mathematical proof of non-commutativity for the README
    comm_check = R @ S - S @ R
    print(f"Commutator [R, S] Norm: {np.linalg.norm(comm_check):.6f}")