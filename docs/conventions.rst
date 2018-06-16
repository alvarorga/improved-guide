===========
Conventions
===========

This is a list of conventions used throughout the repository when
translating matrix product states concepts into real Python code.

Index order in tensors
----------------------

- A ket operator written in MPS form will be composed of several
  rank-three tensors with the virtual index looking upwards. For
  example, the tensor :math:`M` looks like

  .. image:: images/three_legged_tensor.svg
     :align: center

  and its indices are ordered in the `Mps` class as `M[i, s, j]`.

- Left and rigth fixed points are rank-two tensors that look like

  .. image:: images/left_right_fixed_points.svg
     :align: center

  Their indices are accesed as `L[i, j]` and `R[i, j]`.
