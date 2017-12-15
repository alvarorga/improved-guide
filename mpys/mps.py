# MPS class.


class Mps(object):
    """Class for matrix product states (MPS).

    Attributes:
        L (int): length of the MPS.
        D (int): maximum bond dimension.
        M (list[arrays]): list with the tensors that define the MPS in
            each site.
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
