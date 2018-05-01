"""Mps operations. Only between MPSs."""

import numpy as np

import sys
print(sys.path)
from mps import Mps


def contract(psi, phi):
    """Compute the expected value of <psi|phi>."""
    if (not isinstance(psi, Mps)) or (not isinstance(phi, Mps)):
        raise TypeError('The inputs are not MPS.')
    if (psi.L != phi.L) or (psi.d != phi.d):
        raise ValueError('The input MPS do not have matching size '
                         + 'or dimension.')
    # Left tensor that will carry the result of the contraction.
    L = np.eye(2, dtype=np.float64)
    for i in range(psi.L):
        L = np.einsum('im,mjk->ijk', L, psi.A[i])
        L = np.einsum('mni,mnj->ij', L, phi.A[i])
    return np.trace(L)
