"""Tests for the MPS class."""

import sys
import unittest
import numpy as np
sys.path.append('../mpys')
from mps import Mps

def check_left_canonical(psi):
    """Check that psi is left canonical."""
    for A in psi.A:
        rdim = np.shape(A)[2]
        L = np.einsum('mni,mnj->ij', A, A)
        if not np.allclose(L, np.eye(rdim)):
            return False
    else:
        return True

def check_right_canonical(psi):
    """Check that psi is right canonical."""
    for B in reversed(psi.B):
        rdim = np.shape(B)[0]
        R = np.einsum('imn,jmn->ij', B, B)
        if not np.allclose(R, np.eye(rdim)):
            return False
    else:
        return True

class MPSCreationTestCase(unittest.TestCase):
    """Tests for MPS creation routines."""

    def test_creation_GHZ(self):
        """Test the creation of a GHZ state."""
        L = 6
        psi = Mps(L, 'GHZ')
        self.assertEqual(psi.L, 6)
        self.assertEqual(psi.d, 2)
        self.assertEqual(psi.D, 2)
        M = np.zeros((2, 2, 2))
        M[0, 0, 0] = 1
        M[1, 1, 1] = 1
        Ai = np.zeros((1, 2, 2))
        Ai[0, 0, 0] = 1
        Ai[0, 1, 1] = 1
        Bi = np.zeros((1, 2, 2))
        Bi[0, 0, 0] = 1/np.sqrt(2)
        Bi[0, 1, 1] = 1/np.sqrt(2)
        Af = np.zeros((2, 2, 1))
        Af[0, 0, 0] = 1/np.sqrt(2)
        Af[1, 1, 0] = 1/np.sqrt(2)
        Bf = np.zeros((2, 2, 1))
        Bf[0, 0, 0] = 1
        Bf[1, 1, 0] = 1
        self.assertTrue(np.allclose(psi.A[0], Ai))
        self.assertTrue(np.allclose(psi.A[1], M))
        self.assertTrue(np.allclose(psi.A[3], M))
        self.assertTrue(np.allclose(psi.A[5], Af))
        self.assertTrue(np.allclose(psi.B[0], Bi))
        self.assertTrue(np.allclose(psi.B[2], M))
        self.assertTrue(np.allclose(psi.B[5], Bf))
        # Test canonicity.
        # self.assertTrue(check_left_canonical(psi))
        # self.assertTrue(check_right_canonical(psi))

    def test_creation_AKLT(self):
        """Test the creation of an AKLT state."""
        L = 5
        psi = Mps(L, 'AKLT')
        self.assertEqual(psi.L, 5)
        self.assertEqual(psi.d, 3)
        self.assertEqual(psi.D, 2)
        M = np.zeros((2, 3, 2))
        M[1, 0, 0] = -np.sqrt(2/3)
        M[0, 1, 0] = -np.sqrt(1/3)
        M[1, 1, 1] = np.sqrt(1/3)
        M[0, 2, 1] = np.sqrt(2/3)
        Ai = np.zeros((1, 3, 2))
        Ai[0, 0, 0] = 1
        Ai[0, 2, 1] = 1
        Bi = np.zeros((1, 3, 2))
        Bi[0, 0, 0] = 1/np.sqrt(2)
        Bi[0, 2, 1] = 1/np.sqrt(2)
        Af = np.zeros((2, 3, 1))
        Af[0, 0, 0] = 1/np.sqrt(2)
        Af[1, 2, 0] = 1/np.sqrt(2)
        Bf = np.zeros((2, 3, 1))
        Bf[0, 0, 0] = 1
        Bf[1, 2, 0] = 1
        self.assertTrue(np.allclose(psi.A[0], Ai))
        self.assertTrue(np.allclose(psi.A[1], M))
        self.assertTrue(np.allclose(psi.A[3], M))
        self.assertTrue(np.allclose(psi.A[4], Af))
        self.assertTrue(np.allclose(psi.B[0], Bi))
        self.assertTrue(np.allclose(psi.B[2], M))
        self.assertTrue(np.allclose(psi.B[3], M))
        self.assertTrue(np.allclose(psi.B[4], Bf))
        # Test canonicity.
        self.assertTrue(check_left_canonical(psi))
        self.assertTrue(check_right_canonical(psi))

    def test_creation_random(self):
        """Test the creation of a random state."""
        psi = Mps(5, 'random', d=3)
        self.assertTrue(psi.L, 5)
        self.assertTrue(psi.d, 3)
        self.assertTrue(psi.D, 2)
        # Test canonicity.
        # self.assertTrue(check_left_canonical(psi))
        # self.assertTrue(check_right_canonical(psi))
