"""MPS class."""

import numpy as np
from scipy.linalg import qr, rq

class Mps(object):
    """Class for matrix product states (MPS).

    Attributes:
        L (int): length of the MPS.
        d (int): physical dimension.
        D (int): maximum bond dimension.
        A (list of ndarrays): left canonical tensors that define the MPS
            at each site.
        B (list of ndarrays): right canonical tensors that define the
            MPS at each site.

    """

    def __init__(self, L, name=None, d=None):
        """Initialize the MPS class.

        Args:
            L (int): length of the MPS.
            name (str, opt): name of the state that we initialize.
                Examples are: 'GHZ', 'AKLT', and 'random'.
            d (int, opt): physical dimension (only for the random case).
        """
        self.L = L
        self.A = []
        self.B = []
        if name == 'GHZ':
            self.d = 2
            self.D = 2
            M = np.zeros((2, 2, 2), np.float64)
            M[0, 0, 0] = 1
            M[1, 1, 1] = 1
            Ai = np.zeros((1, 2, 2), np.float64)
            Ai[0, 0, 0] = 1
            Ai[0, 1, 1] = 1
            Bi = np.zeros((1, 2, 2), np.float64)
            Bi[0, 0, 0] = 1/np.sqrt(2)
            Bi[0, 1, 1] = 1/np.sqrt(2)
            Af = np.zeros((2, 2, 1), np.float64)
            Af[0, 0, 0] = 1/np.sqrt(2)
            Af[1, 1, 0] = 1/np.sqrt(2)
            Bf = np.zeros((2, 2, 1), np.float64)
            Bf[0, 0, 0] = 1
            Bf[1, 1, 0] = 1
            for i in range(L):
                if i == 0:
                    self.A.append(Ai)
                    self.B.append(Bi)
                elif i == L-1:
                    self.A.append(Af)
                    self.B.append(Bf)
                else:
                    self.A.append(M)
                    self.B.append(M)

        elif name == 'AKLT':
            self.d = 3
            self.D = 2
            M = np.zeros((2, 3, 2), np.float64)
            M[1, 0, 0] = -np.sqrt(2/3)
            M[0, 1, 0] = -np.sqrt(1/3)
            M[1, 1, 1] = np.sqrt(1/3)
            M[0, 2, 1] = np.sqrt(2/3)
            Ai = np.zeros((1, 3, 2), np.float64)
            Ai[0, 0, 0] = 1
            Ai[0, 2, 1] = 1
            Bi = np.zeros((1, 3, 2), np.float64)
            Bi[0, 0, 0] = 1/np.sqrt(2)
            Bi[0, 2, 1] = 1/np.sqrt(2)
            Af = np.zeros((2, 3, 1), np.float64)
            Af[0, 0, 0] = 1/np.sqrt(2)
            Af[1, 2, 0] = 1/np.sqrt(2)
            Bf = np.zeros((2, 3, 1), np.float64)
            Bf[0, 0, 0] = 1
            Bf[1, 2, 0] = 1
            for i in range(L):
                if i == 0:
                    self.A.append(Ai)
                    self.B.append(Bi)
                elif i == L-1:
                    self.A.append(Af)
                    self.B.append(Bf)
                else:
                    self.A.append(M)
                    self.B.append(M)

        elif name == 'random':
            if d is None:
                d = 2
            self.d = d
            self.D = 2

            # Create a left-canonical MPS.
            for i in range(L):
                if i == 0:
                    shape = (1, d, 2)
                elif i == L-1:
                    shape = (2, d, 1)
                else:
                    shape = (2, d, 2)
                M = np.random.rand(*shape)
                M = np.reshape(M, (shape[0]*shape[1], shape[2]))
                Q, _ = qr(M, mode='economic')
                Q = np.reshape(Q, shape)
                self.A.append(Q)

            # Write the MPS in right-canonical form.
            aux_B = []
            R = np.ones((1, 1))
<<<<<<< HEAD
            for i in reversed(range(L)):
=======
            for i in range(L-1, -1, -1):
>>>>>>> 1b0c99c20deb7a920faf5991f4fbf79ba1f1572e
                if i == 0:
                    shape = (1, d, 2)
                elif i == L-1:
                    shape = (2, d, 1)
                else:
                    shape = (2, d, 2)
                M = self.A[i]
                M = np.einsum('isj,jk->isk', M, R)
                M = np.reshape(M, (shape[0], shape[1]*shape[2]))
                R, Q = rq(M, mode='economic')
                Q = np.reshape(Q, shape)
                aux_B.append(Q)
            # Reorder elements in B.
            for i in range(L):
                self.B.append(aux_B[L-1-i])

        else:
            raise NameError('The name of the state was not found.')
