"""Mps operations. Only between MPSs."""

import numpy as np


def contract(psi, phi, optimize=True):
    """Compute the expected value of <psi|phi>.

    Args:
        psi (Mps): bra state.
        phi (Mps): ket state.
        optimize (bool, opt): True if we want the contractions in
            'np.einsum' to run faster. True may have more memory cost.

    Returns:
        (float): the expected value of the contracion <psi|phi>.

    """
    if (psi.L != phi.L) or (psi.d != phi.d):
        raise ValueError('The input MPS do not have matching size '
                         + 'or dimension.')

    # Left tensor that will carry the result of the contraction.
    L = np.eye(1, dtype=np.float64)
    for i in range(psi.L):
        L = np.einsum('mn,mli,nlj->ij', L, psi.A[i], phi.A[i],
                      optimize=optimize)
    return np.trace(L)
