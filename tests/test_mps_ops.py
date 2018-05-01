"""Tests for the MPS operation functions."""

import sys
import unittest
sys.path.append('../mpys')
from mps import Mps

class MPSContractionTestCase(unittest.TestCase):
    """Test MPS contraction."""

    def test_norm_of_states(self):
        """Test that the norm of some states is 1."""
        psi = Mps(6, 'GHZ')
        self.assertAlmostEqual(psi.norm(), 1)
        psi = Mps(7, 'AKLT')
        self.assertAlmostEqual(psi.norm(), 1)
        psi = Mps(5, 'random', d=4)
        self.assertAlmostEqual(psi.norm(), 1)
