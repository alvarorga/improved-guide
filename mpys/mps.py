# MPS class.


class Mps(object):
    r"""Class for matrix product states (MPS).

    Every state can be expressed as:
    .. math::

       | \Psi \rangle = \sum_{i_1, i_2, \cdot, i_L} A^{i_1} A^{i_2}
       \cdot A^{i_{j-1}} M^{i_j} B^{i_{j+1}} \cdot B^{i_L} | i_1 i_2
       \cdot i_L \rangle.

    Attributes:
        L (int): length of the MPS.
        D (int): maximum bond dimension.
        A (array list): left canonical tensors that define the MPS at
            each site.
        B (array list): right canonical tensors that define the MPS at
            each site.
        M (array list): mixed canonical tensors at each site.
    """
    def __init__(self, L, name='None'):
        """Initialize the MPS class.

        Args:
            L (int): length of the MPS.
            name (str): name of the state that we initialize. Examples
                are: 'GHZ' and 'AKLT'.
        """
        pass

    def truncate(self, D_new):
        """Truncate the MPS to a maximum bond dimension D_new.

        Args:
            D_new (int): maximum bond dimension of the truncated MPS.
        """
        pass
