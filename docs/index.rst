vectfit
=========

This implements the Fast Relaxed Vector Fitting algorithm in C++.

The `vectfit` function approximates a response :math:`\mathbf{f}(s)` (generally
a vector) with a rational function:

.. math::
    :label: vf

    \mathbf{f}(s) \approx \sum_{j=1}^{N} \frac{\mathbf{r}_j}{s-p_j} + \sum_{n=0}^{Nc} \mathbf{c}_n s^{n}

where :math:`p_j` and :math:`r_j` are poles and residues in the complex plane
and :math:`r_j` are the polynomial coefficients.

The identification is done using the pole relocating method known as Vector
Fitting [1]_ with relaxed non-triviality constraint for faster convergence
and smaller fitting errors [2]_, and utilization of matrix structure for fast
solution of the pole identifion step [3]_.

More details about the algorithm can be found in this website_.


Contents:

.. toctree::
   :maxdepth: 1

   vectfit 


.. [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
   domain responses by Vector Fitting", IEEE Trans. Power Delivery,
   vol. 14, no. 3, pp. 1052-1061, July 1999.

.. [2] B. Gustavsen, "Improving the pole relocating properties of vector
   fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
   July 2006.

.. [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
   "Macromodeling of Multiport Systems Using a Fast Implementation of
   the Vector Fitting Method", IEEE Microwave and Wireless Components
   Letters, vol. 18, no. 6, pp. 383-385, June 2008.

.. _website: http://www.sintef.no/Projectweb/VECTFIT
