"""MPS class."""

import numpy as np
from scipy.linalg import qr, rq

from mpys.mps_ops import contract


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
        """Initialize an MPS object.

        Args:
            L (int): length of the MPS.
            name (str, opt): name of the state that we initialize.
                Examples are: 'GHZ', 'AKLT', 'random', 'pairs', and
                'mixed'.
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
            self.B = []
            R = np.ones((1, 1))
            for i in reversed(range(L)):
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
                self.B.append(Q)
            self.B.reverse()

        elif name == 'pairs':
            self.d = 2
            self.D = 2
            # Left canonical tensors.
            # Even sites.
            Me = np.zeros((1, 2, 2), np.float64)
            Me[0, 0, 0] = 1
            Me[0, 1, 1] = 1
            # Odd sites.
            Mo = np.zeros((2, 2, 1), np.float64)
            Mo[1, 0, 0] = np.sqrt(1/2)
            Mo[0, 1, 0] = np.sqrt(1/2)
            # Unpaired sites.
            Mu = np.zeros((1, 2, 1), np.float64)
            Mu[0, 0, 0] = 1/np.sqrt(2)
            Mu[0, 1, 0] = 1/np.sqrt(2)
            for i in range(L):
                if i%2 == 1:  # Odd sites.
                    self.A.append(Mo)
                else:  # Even sites.
                    if i != L-1:
                        self.A.append(Me)
                    else:
                        self.A.append(Mu)
            # Right canonical tensors.
            # Even sites.
            Me = np.zeros((1, 2, 2), np.float64)
            Me[0, 0, 0] = np.sqrt(1/2)
            Me[0, 1, 1] = np.sqrt(1/2)
            # Odd sites.
            Mo = np.zeros((2, 2, 1), np.float64)
            Mo[1, 0, 0] = 1
            Mo[0, 1, 0] = 1
            # Unpaired sites.
            Mu = np.zeros((1, 2, 1), np.float64)
            Mu[0, 0, 0] = 1/np.sqrt(2)
            Mu[0, 1, 0] = 1/np.sqrt(2)
            for i in range(L):
                if i%2 == 1:  # Odd sites.
                    self.B.append(Mo)
                else:  # Even sites.
                    if i != L-1:
                        self.B.append(Me)
                    else:
                        self.B.append(Mu)

        elif name == 'mixed':
            self.d = 2
            self.D = 1
            M = np.zeros((1, 2, 1), np.float64)
            M[0, 0, 0] = 1/np.sqrt(2)
            M[0, 1, 0] = 1/np.sqrt(2)
            for i in range(L):
                self.A.append(M)
                self.B.append(M)

        else:
            raise NameError('The name of the state was not found.')

    def norm(self):
        """Compute the norm of the state."""
        norm = contract(self, self)
        return norm

    def truncate_D(self, D):
        """Truncate the bond dimension a the given one."""
        pass

    def enlarge_D(self, new_D):
        """Enlarge the bond dimension to a new one.

        To enlarge D we add 0's to the tensors of the Mps until we
        reach the required bond dimension.

        Args:
            new_D (int): new bond dimension of the Mps.
        """
        if new_D <= self.D:
            # New bond dimension is smaller than actual bond dimension.
            return

        new_A = []
        new_B = []
        for i in range(self.L):
            shape_A = np.shape(self.A[i])
            shape_B = np.shape(self.B[i])
            ti1 = min(i, self.L-i)
            ti2 = min(i+1, self.L-i-1)
            new_shape_A = (2**ti1 if np.log2(new_D) > ti1 else new_D,
                           self.d,
                           2**ti2 if np.log2(new_D) > ti2 else new_D)
            new_shape_B = (2**ti1 if np.log2(new_D) > ti1 else new_D,
                           self.d,
                           2**ti2 if np.log2(new_D) > ti2 else new_D)
            new_A = np.zeros(new_shape_A, dtype=np.float64)
            new_B = np.zeros(new_shape_B, dtype=np.float64)
            # Write the elts of the old A and B tensors into the new
            # ones and take them out of the lists 'A' and 'B'.
            new_A[:shape_A[0], :, :shape_A[2]] = self.A.pop(i)
            new_B[:shape_B[0], :, :shape_B[2]] = self.B.pop(i)
            self.A.insert(i, new_A)
            self.B.insert(i, new_B)
        self.D = new_D
        return
