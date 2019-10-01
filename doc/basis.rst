..
    : IODATA is an input and output module for quantum chemistry.
    :
    : Copyright (C) 2011-2019 The IODATA Development Team
    :
    : This file is part of IODATA.
    :
    : IODATA is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : IODATA is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --

.. _basis_conventions:

Basis set conventions
#####################

IOData can load molecular orbital coefficients, density matrices and atomic orbital
basis sets from various file formats, and it can also write orbitals and the
basis sets in the Molden format. To achieve an unambiguous numerical
representation of these objects, conventions for the ordering basis functions
(within one shell) and normalization of Gaussian primitives must be fixed.

IOData does not use hard-coded conventions but keeps track of them in attributes
of them in ``IOData.obasis``. This attribute is an instance of the
:py:class:`iodata.basis.MolecularBasis` class, of which the ``conventions`` and
``primitive_normalization`` attributes contain all the relevant information.

For the time being, the ``primitive_normalization`` is always set to ``'L2'``,
meaning that the contraction coefficients assume L2-normalized Gaussian
primitives. However, IOData does *not* enforce normalized contractions.

The first subsection provides a mathematical definition of the Gaussian basis
functions, which is followed by the specification of the ``conventions``
attribute of the ``MolecularBasis`` class.


Gaussian basis functions
========================

IOData supports contracted Gaussian basis functions, which have in general the
following form:

.. math:: b(\mathbf{r}; D_1, \ldots, D_k, P, \alpha_1, \ldots, \alpha_K, \mathbf{r}_A) =
          \sum_{k=1}^K D_k N(\alpha_k, P)
          P(\mathbf{r} - \mathbf{r}_A)
          \exp(-\alpha_k \Vert \mathbf{r} - \mathbf{r}_A \Vert^2)

where :math:`K` is the contraction length, :math:`D_k` is a contraction
coefficient, :math:`N` is a normalization constant, :math:`P` is a Cartesian
polynomial, :math:`\alpha_k` is an exponent and :math:`\mathbf{r}_A` is the
center of the basis function. The summation over :math:`k` is
conventionally called a contraction of *primitive Gaussian basis functions*.
The L2-normalization of each primitive depends on both the polynomial and the
exponent and is defined by the following relation:

.. math:: \int \Bigl\vert N(\alpha_k, P) P(\mathbf{r} - \mathbf{r}_A)
               \exp(-\alpha_k \Vert \mathbf{r} - \mathbf{r}_A \Vert^2)
               \Bigr\vert^2 d\mathbf{r} = 1

Two types of polynomials will be defined below: Cartesian and pure (harmonic)
basis functions.


Cartesian basis functions
-------------------------


When the polynomial consists of a single term as follows:

.. math:: P(x,y,z) = x^{n_x} y^{n_y} z^{n_z}

with :math:`n_x`, :math:`n_y`, :math:`n_z`, zero or positive integer powers, one
speaks of `Cartesian Gaussian basis functions`. One refers to the sum of the
powers as the angular momentum of the Cartesian Gaussian basis.

The normalization constant of a primitive function is:

.. math:: N(\alpha_k, n_x, n_y, n_z) = \sqrt{\frac
        {(2\alpha_k/\pi)^{3/2} (4\alpha_k)^{n_x+n_y+n_z}}
        {(2n_x-1)!! (2n_y-1)!! (2n_z-1)!!}
        }

In practice one combines all basis functions of a given angular momentum (or
algebraic order) into one *shell*. A basis specification typically only mentions
the total angular momentum, and it is assumed that all polynomials of that order
are included in the basis set. The number of basis functions, i.e. the number of
polynomials, for a given angular momentum, :math:`\ell=n_x+n_y+n_z`, is
:math:`(\ell+1)(\ell+2)/2`.


Pure or harmonic basis functions
--------------------------------

When the polynomial is a real regular solid harmonic, one speaks of *pure
Gaussian basis functions*:

.. math::
    P(r,\theta,\phi) = C_{\ell m}(r,\theta,\phi)
    \quad \text{or} \quad
    P(r,\theta,\phi) = S_{\ell m}(r,\theta,\phi)

where :math:`C_{\ell m}` and :math:`S_{\ell m}` are cosine- and sine-like real
regular solid harmonics, defined for :math:`\ell \ge 0` as follows:

.. math::
    C_{\ell 0}(r,\theta,\phi) &=
        R_\ell^0(r,\theta,\phi) \\
    C_{\ell m}(r,\theta,\phi) &=
        \sqrt{2} (-1)^m \operatorname{Re}
        R_\ell^m(\theta,\phi)
        \quad m = 1\ldots \ell \\
    S_{\ell m}(r,\theta,\phi) &=
        \sqrt{2} (-1)^m \operatorname{Im}
        R_\ell^m(\theta,\phi)
        \quad m = 1\ldots \ell

where :math:`R_\ell^m` are the regular solid harmonics, which have in general
complex function values. The factor :math:`(-1)^m` undoes the Condon-Shortley
phase. In these equations, spherical coordinates are used:

.. math::
    x &= R\sin\theta\cos\phi \\
    y &= R\sin\theta\sin\phi \\
    z &= R\cos\theta

The regular solid harmonics are derived from the standard spherical harmonics,
:math:`Y_\ell^m`, as follows:

.. math::
    R_\ell^m(r, \theta, \varphi) &=
        \sqrt{\frac{4\pi}{2\ell+1}} \,
        r^\ell \,
        Y_\ell^m(\theta, \varphi) \\
    &=
        \sqrt{\frac{(\ell-m)!}{(\ell+m)!}} \,
        r^\ell \,
        P_\ell^m(\cos{\theta}) \,
        e^{i m \varphi}

where :math:`P_\ell^m` are the associated Legendre functions. After substituting
this definition of the regular solid harmonics into the real forms, one obtains:

.. math::
    C_{\ell 0}(r,\theta,\phi) & = P_\ell^0(\cos{\theta}) \, r^\ell \\
    C_{\ell m}(r,\theta,\phi) & =
        (-1)^m \sqrt{\frac{2(\ell-m)!}{(\ell+m)!}} \,
        r^\ell \,
        P_\ell^m(\cos{\theta}) \,
        \cos(m \phi)
        \quad m = 1\ldots \ell \\
    S_{\ell m}(r,\theta,\phi) & =
        (-1)^m \sqrt{\frac{2(\ell-m)!}{(\ell+m)!}} \,
        r^\ell \,
        P_\ell^m(\cos{\theta}) \,
        \sin(m \phi)
        \quad m = 1\ldots \ell \\

Also here, the factor :math:`(-1)^m` cancels out the Condon-Shortley phase.
These expressions show that cosine-like functions contain a factor :math:`\cos(m
\phi)`, and similarly the sine-like functions contain a factor
:math:`\sin(m \phi)`. The factor :math:`r^\ell` causes real regular solid
harmonics to be homogeneous Cartesian polynomials, i.e. linear combinations of
the Cartesian polynomials defined in the previous subsection.

Real regular solid harmonics are used because the pure s- and p-type functions
are consistent with their Cartesian counterparts:

.. math::
    C_{00}(x,y,z) & = 1 \\
    C_{10}(x,y,z) & = z \\
    C_{11}(x,y,z) & = x \\
    S_{11}(x,y,z) & = y \\
    \dots &


The normalization constant of a pure Gaussian basis function is:

.. math:: N(\alpha_k, \ell) = \sqrt{\frac
        {(2\alpha_k/\pi)^{3/2} (4\alpha_k)^\ell}
        {(2\ell-1)!!}
        }

In practical applications, all the basis functions of a given angular momentum
are used and grouped into a *shell*. A basis specification typically only
mentions the total angular momentum, and it is assumed that all polynomials of
that order are included in the basis set. The number of basis functions, i.e.
the number of polynomials, for a given angular momentum, :math:`\ell`, is
:math:`2\ell+1`.


The ``conventions`` attribute
=============================


Different file formats supported by IOData have an incompatible ordering of
basis functions within one *shell*. Also the sign conventions may differ from
the definitions given above. The ``conventions`` attribute of
:py:class:`iodata.basis.MolecularBasis` specifies the ordering and sign flips
relative to the above definitions. It is a dictionary,

* whose keys are tuples denoting a shell type ``(angmom, char)`` where
  ``angmom`` is a positive integer denoting the angular momentum and ``char`` is
  either ``'c'`` or ``'p'`` for Cartesian are pure, respectively

* and whose values are lists of `basis function strings`, where each string
  denotes one basis function.

A basis function string has a one-to-one correspondence to the Cartesian or
pure polynomials defined above.

* In case of Cartesian functions, :math:`x^{n_x} y^{n_y} z^{n_z}` is represented
  by the string ``'x' * nx + 'y' * ny + 'z' * nz``, except for the s-type
  function, which is represented by ``'1'``.

* In case of pure functions, :math:`C_{\ell m}` is represented by
  ``'c{}'.format(m)`` and :math:`S_{\ell m}` is by ``'s{}'.format(m)``. The
  angular momentum quantum number is not included because it is implied by the
  key in the ``conventions`` dictionary.

Each basis function string can be prefixed with a minus sign, to denote a
sign flip with respect to the definitions on this page. The order of the string
in the list defines the order of the corresponding basis functions within one
shell.

For example, pure and Cartesian s, p and d functions in Gaussian FCHK files
adhere to the following convention:

.. code-block:: python

    conventions = {
        (0, 'c'): ['1'],
        (1, 'c'): ['x', 'y', 'z'],
        (2, 'c'): ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
        (2, 'p'): ['c0', 'c1', 's1', 'c2', 's2'],
    }

(Pure s and p functions are never used in a Gaussian FCHK file.)


Notes on other conventions
==========================

To avoid confusion, negative magnetic quantum numbers are never used to label
pure functions in IOData. The basis strings contain `'c'` and `'s'` instead. In
the literature, e.g. in the book *Molecular Electronic-Structure Theory* by
Helgaker, Jørgensen and Olsen, negative magnetic quantum numbers for pure
functions are usually referring to sine-like functions:

.. math::
    R_{\ell, m} &= C_{\ell m} \quad m = 0 \ldots \ell \\
    R_{\ell, -m} &= S_{\ell m} \quad m = 1 \ldots \ell

Note that :math:`\ell` and :math:`m` both appear as subscripts in
:math:`R_{\ell, m}` and :math:`R_{\ell, -m}` to tell them apart from their
complex counterparts.


Transformation from Cartesian to pure functions
===============================================

Pure Gaussian primitives can written as linear combinations of Cartesian ones.
Hence, integrals over Cartesian functions can also be transformed
into integrals over pure primitives. This transformation is the last step
in the calculation of the overlap matrix in IOData:

1) Integrals are first computed for Gaussian primitives without normalization.
2) Normalization constants for Cartesian primitives are multiplied into the
   integrals.
3) Integrals over primitives are contracted.
4) Optionally, the integrals for Cartesian functions are transformed into
   integrals for pure functions.

For the last step, pre-computed transformations matrices (generated by
``tools/harmonics.py`` are stored in ``iodata/overlap_cartpure.py`` using the
``HORTON2_CONVENTIONS``. The derivation of these transformation matrices is
explained below.


Recursive computation of real regular solid harmonics
-----------------------------------------------------

First, we construct two sets of recursion relations for :math:`\phi` and
:math:`\theta` separately. These will be combined to form the final set of
recursion relations that directly operate on the real regular solid harmonics.
In these two sets, the notation :math:`\rho = \sqrt{x^2 + y^2}` is used.

The first set of recursion relations starts from a fairly trivial idea:

.. math::
    \begin{split}
        \rho^m [\cos(m\phi) + i\sin(m\phi)]
            &= \rho^m \exp(im\phi) \\
            &= \rho \exp(i\phi) \; \rho^{m-1}\exp(i(m-1)\phi) \\
            &= (x + iy) \; \rho^{m-1} [\cos((m-1)\phi) + i\sin((m-1)\phi)]
    \end{split}

.. math::
    \rho \cos(\phi) &= x \\
    \rho \sin(\phi) &= y \\
    \rho \cos(m\phi) &= x \cos((m-1)\phi) - y \sin((m-1)\phi) \\
    \rho \sin(m\phi) &= x \sin((m-1)\phi) + y \cos((m-1)\phi)

Second, recursion relations for associated Legendre functions can be modified to
contain :math:`r`, :math:`z` and :math:`\rho`, such that :math:`\cos\theta` does
not appear explicitly:

.. math::
    P_0^0(\cos\theta) &= 1 \\
    r^\ell P_\ell^\ell(\cos\theta)
        &= (2\ell - 1) \rho \; r^{\ell-1} P_{\ell-1}^{\ell-1}(\cos\theta) \\
    r^{\ell} P_{\ell}^{\ell-1}(\cos\theta)
        &= -(2\ell - 1) z \; r^{\ell-1} P_{\ell-1}^{\ell-1}(\cos\theta) \\
    r^\ell P_{\ell}^{m}(\cos\theta)
        &= \frac{2\ell - 1}{\ell - m} z \; r^{\ell-1} P_{\ell-1}^{m}(\cos\theta)
           -\frac{\ell + m - 1}{\ell - m} r^2 \; r^{\ell-2} P_{\ell-2}^{m}(\cos\theta)

The two sets could be used separately to construct real regular solid harmonics,
but they feature :math:`\rho=\sqrt{x^2+y^2}`, while the regular solid harmonics
should be homogeneous polynomials. We can get rid of :math:`\rho` by combining
the two sets into one:

.. math::
    C_{0,0} ={}& 1 \\
    C_{1,0} ={}& z \\
    C_{1,1} ={}& x \\
    S_{1,1} ={}& y \\
    C_{\ell,\ell}
        ={}& \sqrt{\frac{2\ell-1}{2\ell}} \;
             \Bigl[x C_{\ell-1,\ell-1} - y S_{\ell-1,\ell-1} \Bigr]
             \quad \forall \; \ell > 1 \\
    S_{\ell,\ell}
        ={}& \sqrt{\frac{2\ell-1}{2\ell}} \;
             \Bigl[x S_{\ell-1,\ell-1} + y C_{\ell-1,\ell-1} \Bigr]
             \quad \forall \; \ell > 1 \\
    \{CS\}_{\ell,\ell-1}
        ={}& z \sqrt{2\ell-1} \;
        \{CS\}_{\ell-1, \ell-1}
        \quad \forall \; \ell > 1 \\
    \{CS\}_{\ell,m}
        ={}& \frac{(2\ell - 1)z}{\sqrt{(\ell+m)(\ell-m)}} \{CS\}_{\ell-1,m} \nonumber \\
           & - r^2 \sqrt{\frac{(\ell - m  - 1)(\ell + m - 1)}{(\ell + m)(\ell - m)}} \{CS\}_{\ell - 2,m} \nonumber \\
           & \quad \forall \; \ell > m + 1 \text{ and } m \ge 0

These equations show that real regular solid harmonics are homogeneous
polynomials in :math:`x`, :math:`y` and :math:`z`. Advantages of this approach
are (i) the absence of trigonometric expressions and (ii) the similarity between
cosine and sine expressions. (Coefficients can be reused.) These recursion
relations should be numerically stable for the computation of real regular solid
harmonics as a function of Cartesian coordinates. They can also be used to build
a transformation matrix from Cartesian mononomials into real regular solid
harmonics.


Transformation matrices without normalization
---------------------------------------------

The above recursion relations result in the following transformation matrices.
These were obtained by running:

.. code-block:: bash

    python tools/harmonics.py none latex 3

.. math::
    \left(\begin{array}{c}
        b(C_{20}) \\ b(C_{21}) \\ b(S_{21}) \\ b(C_{22}) \\ b(S_{22})
    \end{array}\right)
        &=
    \left(\begin{array}{cccccc}
        - \frac{1}{2} & \cdot & \cdot & - \frac{1}{2} & \cdot & 1 \\
        \cdot & \cdot & \sqrt{3} & \cdot & \cdot & \cdot \\
        \cdot & \cdot & \cdot & \cdot & \sqrt{3} & \cdot \\
        \frac{\sqrt{3}}{2} & \cdot & \cdot & - \frac{\sqrt{3}}{2} & \cdot & \cdot \\
        \cdot & \sqrt{3} & \cdot & \cdot & \cdot & \cdot \\
    \end{array}\right)
    \left(\begin{array}{c}
        b(xx) \\ b(xy) \\ b(xz) \\ b(yy) \\ b(yz) \\ b(zz)
    \end{array}\right)
    \\
    \left(\begin{array}{c}
        b(C_{30}) \\ b(C_{31}) \\ b(S_{31}) \\ b(C_{32}) \\ b(S_{32}) \\ b(C_{33}) \\ b(S_{33})
    \end{array}\right)
        &=
    \left(\begin{array}{cccccccccc}
        \cdot & \cdot & - \frac{3}{2} & \cdot & \cdot & \cdot & \cdot & - \frac{3}{2} & \cdot & 1 \\
        - \frac{\sqrt{6}}{4} & \cdot & \cdot & - \frac{\sqrt{6}}{4} & \cdot & \sqrt{6} & \cdot & \cdot & \cdot & \cdot \\
        \cdot & - \frac{\sqrt{6}}{4} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{6}}{4} & \cdot & \sqrt{6} & \cdot \\
        \cdot & \cdot & \frac{\sqrt{15}}{2} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{15}}{2} & \cdot & \cdot \\
        \cdot & \cdot & \cdot & \cdot & \sqrt{15} & \cdot & \cdot & \cdot & \cdot & \cdot \\
        \frac{\sqrt{10}}{4} & \cdot & \cdot & - \frac{3 \sqrt{10}}{4} & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
        \cdot & \frac{3 \sqrt{10}}{4} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{10}}{4} & \cdot & \cdot & \cdot \\
    \end{array}\right)
    \left(\begin{array}{c}
        b(xxx) \\ b(xxy) \\ b(xxz) \\ b(xyy) \\ b(xyz) \\ b(xzz) \\ b(yyy) \\ b(yyz) \\ b(yzz) \\ b(zzz)
    \end{array}\right)





Taking into account normalization
---------------------------------

For the calculation of the overlap matrix, the transformations need to be
modified, to transform normalized Cartesian functions into normalized pure
functions. Accounting for normalization yields slightly different matrices
shown below. These were obtained by running:

.. code-block:: bash

    python tools/harmonics.py L2 latex 3

.. math::
    \left(\begin{array}{c}
        b(C_{20}) \\ b(C_{21}) \\ b(S_{21}) \\ b(C_{22}) \\ b(S_{22})
    \end{array}\right)
        &=
    \left(\begin{array}{cccccc}
        - \frac{1}{2} & \cdot & \cdot & - \frac{1}{2} & \cdot & 1 \\
        \cdot & \cdot & 1 & \cdot & \cdot & \cdot \\
        \cdot & \cdot & \cdot & \cdot & 1 & \cdot \\
        \frac{\sqrt{3}}{2} & \cdot & \cdot & - \frac{\sqrt{3}}{2} & \cdot & \cdot \\
        \cdot & 1 & \cdot & \cdot & \cdot & \cdot \\
    \end{array}\right)
    \left(\begin{array}{c}
        b(xx) \\ b(xy) \\ b(xz) \\ b(yy) \\ b(yz) \\ b(zz)
    \end{array}\right)
    \\
    \left(\begin{array}{c}
        b(C_{30}) \\ b(C_{31}) \\ b(S_{31}) \\ b(C_{32}) \\ b(S_{32}) \\ b(C_{33}) \\ b(S_{33})
    \end{array}\right)
        &=
    \left(\begin{array}{cccccccccc}
        \cdot & \cdot & - \frac{3 \sqrt{5}}{10} & \cdot & \cdot & \cdot & \cdot & - \frac{3 \sqrt{5}}{10} & \cdot & 1 \\
        - \frac{\sqrt{6}}{4} & \cdot & \cdot & - \frac{\sqrt{30}}{20} & \cdot & \frac{\sqrt{30}}{5} & \cdot & \cdot & \cdot & \cdot \\
        \cdot & - \frac{\sqrt{30}}{20} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{6}}{4} & \cdot & \frac{\sqrt{30}}{5} & \cdot \\
        \cdot & \cdot & \frac{\sqrt{3}}{2} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{3}}{2} & \cdot & \cdot \\
        \cdot & \cdot & \cdot & \cdot & 1 & \cdot & \cdot & \cdot & \cdot & \cdot \\
        \frac{\sqrt{10}}{4} & \cdot & \cdot & - \frac{3 \sqrt{2}}{4} & \cdot & \cdot & \cdot & \cdot & \cdot & \cdot \\
        \cdot & \frac{3 \sqrt{2}}{4} & \cdot & \cdot & \cdot & \cdot & - \frac{\sqrt{10}}{4} & \cdot & \cdot & \cdot \\
    \end{array}\right)
    \left(\begin{array}{c}
        b(xxx) \\ b(xxy) \\ b(xxz) \\ b(xyy) \\ b(xyz) \\ b(xzz) \\ b(yyy) \\ b(yyz) \\ b(yzz) \\ b(zzz)
    \end{array}\right)
