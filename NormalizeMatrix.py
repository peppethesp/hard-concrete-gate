import numpy as np

def normalize_matrices(A, B):
    """
    Normalize a matrix for a problem of the type A @ X = B

    This is done in order to use the resulting dataset in a regularization algorithm effectively.
    """
    norms = np.linalg.norm(A, ord=2, axis=0)[None, :]
    A_norm = A/norms

    B_std = np.std(B)
    B_norm = B/B_std

    return A_norm, B_norm, norms, B_std
