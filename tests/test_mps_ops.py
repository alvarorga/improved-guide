"""Tests for the MPS operation functions."""

import os
import sys
import unittest
sys.path.append(os.path.abspath('./mpys'))
from mps import Mps

class MPSContractionTestCase(unittest.TestCase):
    """Test MPS contraction."""

    def test_norm_of_states(self):
        """Test that the norm of some states is 1."""
        self.assertAlmostEqual(Mps(6, 'GHZ').norm(), 1)
        self.assertAlmostEqual(Mps(7, 'AKLT').norm(), 1)
        self.assertAlmostEqual(Mps(5, 'random', d=4).norm(), 1)
