===========
Conventions
===========

This is a list of conventions used throughout the repository when
translating matrix product states concepts into real Python code.

Index order in tensors
----------------------

- **Ket states**, :math:`|\Psi\rangle`, written in MPS form will be
  composed of several 3-rank tensors with the virtual index looking
  upwards. For example, the tensor :math:`M` looks like

  .. image:: images/ket_state_tensor.svg
     :align: center

  and its indices are ordered in the `Mps` class as `M[i, s, j]`.

- **Bra states**, :math:`\langle \Psi|`, written in MPS form will also
  be composed of several 3-rank tensors but with the virtual index
  looking downwards. The tensor :math:`M^\dagger` looks like

  .. image:: images/bra_state_tensor.svg
     :align: center

  and its indices are ordered as `conj(M[i, s, j])`.

- **Left** and **rigth fixed points** are rank-two tensors that look like

  .. image:: images/left_right_fixed_points.svg
     :align: center

  Their indices are accesed as `L[i, j]` and `R[i, j]`.
