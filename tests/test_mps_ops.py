"""Tests for the MPS operation functions."""

import sys
import unittest
import numpy as np

sys.path.append('..')
from mpys.mps import Mps
from mpys.mps_ops import contract


class MPSContractionTestCase(unittest.TestCase):
    """Test MPS contraction."""

    def test_exceptions_of_contract(self):
        """Test the exceptions of the contract function."""
        pass

    def test_contraction_of_two_states(self):
        """Test the contraction of two different Mps."""
        # Pairs and mixed Mps.
        self.assertAlmostEqual(contract(Mps(4, 'pairs'), Mps(4, 'mixed')),
                               np.sqrt(1/2)**2)
        self.assertAlmostEqual(contract(Mps(8, 'pairs'), Mps(8, 'mixed')),
                               np.sqrt(1/2)**4)
        self.assertAlmostEqual(contract(Mps(7, 'pairs'), Mps(7, 'mixed')),
                               np.sqrt(1/2)**3)
        # Pairs and GHZ Mps.
        self.assertAlmostEqual(contract(Mps(4, 'GHZ'), Mps(4, 'pairs')), 0)
        self.assertAlmostEqual(contract(Mps(7, 'GHZ'), Mps(7, 'pairs')), 0)
        # GHZ and mixed Mps.
        self.assertAlmostEqual(contract(Mps(7, 'GHZ'), Mps(7, 'mixed')), 1/8)
        self.assertAlmostEqual(contract(Mps(4, 'GHZ'), Mps(4, 'mixed')),
                               1/np.sqrt(8))
