#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor-blas/xlinalg.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <complex>
#include <cmath>

#include "pybind11/pybind11.h"

using namespace std::complex_literals;

// Complex number zero
constexpr std::complex<double> C_ZERO {0.0, 0.0};

// Tolerances used by relaxed vector fitting
constexpr double TOLlow  = 1e-18;
constexpr double TOLhigh = 1e+18;

//! Multipole formalism evaluation function
//!
//! f(s) = REAL[residues/(s - poles)] + Polynomials(s)
//! Note the input variable s is real and only the real part of the
//! result f is returned.
//!
//! @param s          array of variables to be evaluated. dimension: (Ns)
//! @param poles      poles. dimension: (N)
//! @param residues   residues. dimension: (Nv, N)
//! @param polys      curvefit (Polynomial) coefficients. dimension: (Nv, Nc)
//! @return           f. dimension: (Nv, Ns)

xt::pyarray<double>
evaluate(xt::pyarray<double> s,
         xt::pyarray<std::complex<double>> poles,
         xt::pyarray<std::complex<double>> residues,
         xt::pyarray<double> polys = (xt::pyarray<double>) {})
{
  // Check input arguments
  // s
  if (s.dimension() != 1)
  {
    throw std::invalid_argument("Error: input s is not 1-dimensional.");
  }
  auto Ns = s.size();

  // poles
  if (poles.dimension() != 1)
  {
    throw std::invalid_argument("Error: input poles is not 1-dimensional.");
  }
  auto N = poles.size();

  // residues
  // convert the residues to be two dimensional if it is one dimensional
  if (residues.dimension() == 1 && residues.size() == N)
  {
    residues.reshape({1, N});
  }
  else if (residues.dimension() != 2)
  {
    throw std::invalid_argument("Error: input residues is not 2-dimensional.");
  }
  else if (residues.shape()[1] != N)
  {
    throw std::invalid_argument("Error: 2nd dimension of residues does not "
                                "match the length of poles.");
  }
  auto Nv = residues.shape()[0];

  // polynomial coefficients
  // none polynomials by default
  if (polys.dimension() == 0 || xt::all(xt::equal(polys, 0.)))
  {
    polys = xt::zeros<double>({Nv, (size_t)0});
  }
  else if (polys.dimension() == 1 && Nv == 1)
  {
    polys.reshape({1, polys.size()});
  }
  else if (polys.dimension() !=2)
  {
    throw std::invalid_argument("Error: input polys is not 2-dimensional.");
  }
  else if (polys.shape()[0] != Nv)
  {
    throw std::invalid_argument("Error: 1st dimension of polys does not "
                                "match the 1st dimension of residues.");
  }
  auto Nc = polys.shape()[1];

  // Initialize results array
  xt::pyarray<double> f({Nv, Ns}, 0.0);

  // Evaluate the multipole form
  size_t m, n;
  xt::xtensor<std::complex<double>, 2> Dk2({Ns, N}, C_ZERO);
  for (m = 0; m < N; m++)
  {
    xt::view(Dk2, xt::all(), m) = 1.0 / (s - poles(m));
  }
  for (n = 0; n < Nv; n++)
  {
    xt::view(f, n) = xt::real(xt::linalg::dot(Dk2,
                    xt::xarray<std::complex<double>>(xt::view(residues, n))));
    for (m = 0; m < Nc; m++)
    {
      xt::view(f, n) += xt::pow(s, m) * polys(n, m);
    }
  }

  // Return
  return f;
}


//! Fast Relaxed Vector Fitting function
//!
//! Approximate f(s) with a rational function:
//!         f(s)=R*(s*I-A)^(-1) + Polynomials*s
//! where f(s) is a vector of elements.
//! When f(s) is a vector, all elements become fitted with a common pole set.
//! The identification is done using the pole relocating method known as Vector
//! Fitting [1] with relaxed non-triviality constraint for faster convergence
//! and smaller fitting errors [2], and utilization of matrix structure for fast
//! solution of the pole identifion step [3].
//! [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
//!     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
//!     vol. 14, no. 3, pp. 1052-1061, July 1999.
//! [2] B. Gustavsen, "Improving the pole relocating properties of vector
//!     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
//!     July 2006.
//! [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
//!     "Macromodeling of Multiport Systems Using a Fast Implementation of
//!     the Vector Fitting Method", IEEE Microwave and Wireless Components
//!     Letters, vol. 18, no. 6, pp. 383-385, June 2008.
//! All credit goes to Bjorn Gustavsen for his MATLAB implementation.
//! (http://www.sintef.no/Projectweb/VECTFIT/)
//!
//! @param f          function (vector) to be fitted. dimension: (Nv, Ns)
//! @param s          vector of sample points. dimension: (Ns)
//! @param poles      vector of initial poles. dimension: (N)
//! @param weight     the system matrix is weighted using this array
//! @param n_polys    Nc: Number of curvefit (Polynomial) coefficients, [0, 11]
//! @param skip_pole  if the pole identification part is skipped
//! @param skip_res   if the residue identification part is skipped
//! @return           Tuple(poles, residues, polys, fit, rmserr)

std::tuple<xt::pyarray<std::complex<double>>,
           xt::pyarray<std::complex<double>>,
           xt::pyarray<double>,
           xt::pyarray<double>,
           double>
vectfit(xt::pyarray<double> &f,
        xt::pyarray<double> &s,
        xt::pyarray<std::complex<double>> &poles,
        xt::pyarray<double> &weight,
        int n_polys = 0,
        bool skip_pole = false,
        bool skip_res = false)
{
  // Check input arguments
  if (f.dimension() != 2)
  {
    throw std::invalid_argument("Error: input f is not 2-dimensional.");
  }
  auto shape = f.shape();
  auto Nv = shape[0];
  auto Ns = shape[1];
  if (s.dimension() != 1)
  {
    throw std::invalid_argument("Error: input s is not 1-dimensional.");
  }
  if (Ns != s.size())
  {
    throw std::invalid_argument("Error: 2nd dimension of f does not match the "
                                "length of s.");
  }
  auto N = poles.size();
  if (f.shape() != weight.shape())
  {
    throw std::invalid_argument("Error: shape of weight does not match shape of"
                                " f.");
  }
  size_t Nc = (size_t)n_polys;
  if (Nc < 0 || Nc > 11)
  {
    throw std::invalid_argument("Error: input n_polys is not in range [0, 11].");
  }

  // Initialize arrays
  xt::pyarray<std::complex<double>> residues({Nv, N}, C_ZERO); // residues (R)
  xt::pyarray<double> polys({Nv, Nc}, 0.0); // polynomial coefficients (P)
  xt::pyarray<double> fit({Nv, Ns}, 0.0); // fitted signals on s
  double rmserr = 0.0; // RMS error between f and fit

  // If 0 poles and 0 cf order, return
  if (N == 0 && Nc == 0)
  {
    rmserr = xt::linalg::norm(f) / std::sqrt(Nv * Ns);
    return std::make_tuple(poles, residues, polys, fit, rmserr);
  }

  // Iteration variables
  size_t m, n;

  //============================================================================
  // POLE IDENTIFICATION
  //============================================================================
  if (!skip_pole && N > 0)
  {
    // Finding out which starting poles are complex
    xt::xtensor<int, 1> cindex({N}, 0); // 0-real, 1-complex, 2-conjugate
    for (m = 0; m < N; m++)
    {
      if (std::imag(poles(m)) != 0.0)
      {
        if ((m == 0) || ((m > 0) && (cindex(m - 1) == 0 || cindex(m - 1) == 2)))
        {
          if (m >= N - 1 || std::conj(poles(m)) != poles(m + 1))
          {
            throw std::invalid_argument("Error: complex poles are not conjugate"
                                        " pairs.");
          }
          cindex(m) = 1;
          cindex(m + 1) = 2;
          m++;
        }
      }
    }

    // Building system - matrixes
    xt::xtensor<std::complex<double>, 2> Dk({Ns, N + std::max(Nc, (size_t)1)},
                                            C_ZERO);
    for (m = 0; m < N; m++)
    {
      auto p = poles(m);
      if (cindex(m) == 0) // real pole
        xt::view(Dk, xt::all(), m) = 1. / (s - p);
      else if (cindex(m) == 1) // complex pole, 1st
        xt::view(Dk, xt::all(), m) = 1. / (s - p) + 1. / (s - std::conj(p));
      else if (cindex(m) == 2) // complex pole, 2nd
        xt::view(Dk, xt::all(), m) = 1i / (s -  std::conj(p)) - 1i / (s - p);
      else
        throw std::runtime_error("Error: unknown cindex value.");
    }

    // Check infinite values
    xt::filter(Dk, xt::isinf(Dk)) = TOLhigh + 0.0i;

    // Polynomial terms
    xt::view(Dk, xt::all(), N) = std::complex<double>(1.0, 0.0);
    for (m = 1; m < Nc; m++)
    {
      xt::view(Dk, xt::all(), N + m) = xt::pow(s, m) + 0.0i;
    }

    // Scaling for last row of LS-problem (pole identification)
    double scale = 0.0;
    for (m = 0; m < Nv; m++)
    {
      scale += std::pow(xt::linalg::norm(xt::view(weight, m) * xt::view(f, m)),
                        2);
    }
    scale = std::sqrt(scale) / Ns;

    // A matrix
    xt::xtensor<double, 2> AA({Nv * (N + 1), N + 1}, 0.0);
    xt::xtensor<double, 1> bb({Nv * (N + 1)}, 0.0);
    for (n = 0; n < Nv; n++)
    {
      xt::xtensor<std::complex<double>, 2> A1({Ns, N + Nc + N + 1}, C_ZERO);
      // left block
      for (m = 0; m < N + Nc; m++)
      {
        xt::view(A1, xt::all(), m) = xt::view(weight, n) *
                                     xt::view(Dk, xt::all(), m);
      }
      // right block
      for (m = 0; m < N + 1; m++)
      {
        xt::view(A1, xt::all(), N + Nc + m) = -xt::view(weight, n) *
                                    xt::view(Dk, xt::all(), m) * xt::view(f, n);
      }

      xt::xtensor<double, 2> A({2 * Ns + 1, N + Nc + N + 1}, 0.0);
      xt::view(A, xt::range(0, Ns)) = xt::real(A1);
      xt::view(A, xt::range(Ns, 2 * Ns)) = xt::imag(A1);

      // Integral criterion for sigma
      if (n == Nv - 1)
      {
        for (m = 0; m < N + 1; m++)
        {
          auto d = xt::sum(xt::view(Dk, xt::all(), m))();
          A(2 * Ns, N + Nc + m) = std::real(scale * d);
        }
      }

      // QR decomposition
      // Hotspots of the algorithm
      auto QR_tuple = xt::linalg::qr(A);
      auto R = std::get<1>(QR_tuple);
      xt::view(AA, xt::range(n*(N+1), (n+1)*(N+1))) =
            xt::view(R, xt::range(N+Nc, N+Nc+N+1), xt::range(N+Nc, N+Nc+N+1));
      if (n == Nv - 1)
      {
        auto Q = std::get<0>(QR_tuple);
        xt::view(bb, xt::range(n*(N+1), (n+1)*(N+1))) = Ns * scale *
              xt::view(Q, Q.shape()[0] - 1, xt::range(N+Nc, Q.shape()[1]));
      }
    }

    xt::xtensor<double, 1> Escale({N + 1}, 0.0);
    for (m = 0; m < N + 1; m++)
    {
      Escale(m) = 1.0 / xt::linalg::norm(xt::view(AA, xt::all(), m));
      xt::view(AA, xt::all(), m) *= Escale(m);
    }

    auto results = xt::linalg::lstsq(AA, bb);
    auto x = std::get<0>(results);
    x *= Escale;
    auto C = xt::xarray<double>(xt::view(x, xt::range(0, x.size() - 1)));
    auto D = x(x.size() - 1);

    // Situation: produced D of sigma extremely is small or large
    // Solve again, without relaxation
    if (std::abs(D) < TOLlow || std::abs(D) > TOLhigh)
    {
      xt::xtensor<double, 2> AA({Nv * N, N}, 0.0);
      xt::xtensor<double, 1> bb({Nv * N}, 0.0);
      if (x(x.size() - 1) == 0.0)
      {
        D = 1.0;
      }
      else if (std::abs(x(x.size() - 1)) < TOLlow)
      {
        D = x(x.size() - 1) > 0 ? TOLlow : -TOLlow;
      }
      else if (std::abs(x(x.size() - 1)) > TOLhigh)
      {
        D = x(x.size() - 1) > 0 ? TOLhigh : -TOLhigh;
      }

      for (n = 0; n < Nv; n++)
      {
        xt::xtensor<std::complex<double>, 2> A1({Ns, N + Nc + N}, C_ZERO);
        for (m = 0; m < N + Nc; m++)
        {
          xt::view(A1, xt::all(), m) = xt::view(weight, n) *
                                       xt::view(Dk, xt::all(), m);
        }
        for (m = 0; m < N; m++)
        {
          xt::view(A1, xt::all(), N + Nc + m) = -xt::view(weight, n) *
                                    xt::view(Dk, xt::all(), m) * xt::view(f, n);
        }
        auto A = xt::xarray<double>(xt::concatenate(xt::xtuple(xt::real(A1),
                                                    xt::imag(A1))));
        auto b1 = D * xt::view(weight, n) * xt::view(f, n);
        auto b = xt::xarray<double>(xt::concatenate(xt::xtuple(xt::real(b1),
                                                    xt::imag(b1))));

        // QR decomposition
        auto QR_tuple = xt::linalg::qr(A);
        auto Q = std::get<0>(QR_tuple);
        auto R = std::get<1>(QR_tuple);
        xt::view(AA, xt::range(n*N, (n+1)*N)) =
            xt::view(R, xt::range(N+Nc, N+Nc+N), xt::range(N+Nc, N+Nc+N));
        xt::view(bb, xt::range(n*N, (n+1)*N)) = xt::linalg::dot(
            xt::transpose(xt::view(Q, xt::all(), xt::range(N+Nc, N+Nc+N))), b);
      }

      xt::xtensor<double, 1> Escale ({N}, 0.0);
      for (m = 0; m < N; m++)
      {
        xt::view(Escale, m) = 1.0 / xt::linalg::norm(xt::view(AA, xt::all(), m));
        xt::view(AA, xt::all(), m) *= Escale(m);
      }

      auto results = xt::linalg::lstsq(AA, bb);
      auto C = std::get<0>(results);
      C *= Escale;
    }

    // We now calculate the zeros for sigma
    xt::xtensor<double, 2> LAMBD({N, N}, 0.0);
    xt::xtensor<double, 2> SERB({N, (size_t)1}, 1.0);
    for (m = 0; m < N; m++)
    {
      if (cindex(m) == 0) // real pole
      {
        LAMBD(m, m) = std::real(poles(m));
      }
      else if (cindex(m) == 1)
      {
        auto x = std::real(poles(m));
        auto y = std::imag(poles(m));
        LAMBD(m, m) = x;
        LAMBD(m + 1, m + 1) = x;
        LAMBD(m + 1, m) = -y;
        LAMBD(m, m + 1) = y;
        SERB(m, 1) = 2.0;
        SERB(m + 1, 1) = 0.0;
      }
    }

    // Update poles
    C.reshape({1, N});
    auto ZER = LAMBD - xt::linalg::dot(SERB, C) / D;
    poles = xt::linalg::eigvals(ZER);
  }

  //============================================================================
  //  RESIDUE IDENTIFICATION
  //============================================================================
  if (!skip_res)
  {
    // Finding out which poles are complex:
    xt::xtensor<int, 1> cindex({N}, 0); // 0-real, 1-complex, 2-conjugate
    for (m = 0; m < N; m++)
    {
      if (std::imag(poles(m)) != 0.0)
      {
        if ((m == 0) || ((m > 0) && (cindex(m - 1) == 0 || cindex(m - 1) == 2)))
        {
          if (m >= N - 1 || std::conj(poles(m)) != poles(m + 1))
          {
            throw std::invalid_argument("Error: complex poles are not conjugate"
                                        " pairs.");
          }
          cindex(m) = 1;
          cindex(m + 1) = 2;
          m++;
        }
      }
    }

    // Calculate the SER for f (new fitting), using the above calculated
    // zeros as known poles
    xt::xtensor<std::complex<double>, 2> Dk({Ns, N + Nc}, C_ZERO);
    for (m = 0; m < N; m++)
    {
      auto p = poles(m);
      if (cindex(m) == 0) // real pole
        xt::view(Dk, xt::all(), m) = 1. / (s - p);
      else if (cindex(m) == 1) // complex pole, 1st
        xt::view(Dk, xt::all(), m) = 1. / (s - p) + 1. / (s - std::conj(p));
      else if (cindex(m) == 2) // complex pole, 2nd
        xt::view(Dk, xt::all(), m) = 1i / (s - std::conj(p)) - 1i / (s - p);
      else
        throw std::runtime_error("Error: unknown cindex value.");
    }
    for (m = 0; m < Nc; m++)
    {
      xt::view(Dk, xt::all(), N + m) = xt::pow(s, m) + 0.0i;
    }

    xt::xtensor<double, 2> Cr({Nv, N}, 0.0);
    for (n = 0; n < Nv; n++)
    {
      xt::xtensor<std::complex<double>, 2> A1({Ns, N + Nc}, C_ZERO);
      for (m = 0; m < N + Nc; m++)
      {
        xt::view(A1, xt::all(), m) = xt::view(weight, n) *
                                     xt::view(Dk, xt::all(), m);
      }
      auto A = xt::xtensor<double, 2>(xt::concatenate(xt::xtuple(xt::real(A1),
                                                      xt::imag(A1))));
      auto b1 = xt::view(weight, n) * xt::view(f, n);
      auto b = xt::xtensor<double, 1>(xt::concatenate(xt::xtuple(xt::real(b1),
                                                      xt::imag(b1))));

      xt::xtensor<double, 1> Escale({N + Nc}, 0.0);
      for (m = 0; m < N + Nc; m++)
      {
        xt::view(Escale, m) = 1.0 / xt::linalg::norm(xt::view(A, xt::all(), m));
        xt::view(A, xt::all(), m) *= Escale(m);
      }

      auto results = xt::linalg::lstsq(A, b);
      auto x = std::get<0>(results);
      x *= Escale;

      xt::view(Cr, n) = xt::view(x, xt::range(0, N));

      if (Nc > 0)
      {
        xt::view(polys, n) = xt::view(x, xt::range(N, N + Nc));
      }
    }

    // Get complex residues
    for (m = 0; m < N; m++)
    {
      if (cindex(m) == 0)
      {
        for (n = 0; n < Nv; n++)
        {
          residues(n, m) = std::complex<double>(Cr(n, m));
        }
      }
      else if (cindex(m) == 1)
      {
        for (n = 0; n < Nv; n++)
        {
          auto r1 = Cr(n, m);
          auto r2 = Cr(n, m + 1);
          residues(n, m) = r1 + 1i * r2;
          residues(n, m + 1) = r1 - 1i * r2;
        }
      }
    }

    // Calculate fit on s
    fit = evaluate(s, poles, residues, polys);

    // RMS error
    rmserr = xt::linalg::norm(fit - f) / std::sqrt(Nv * Ns);
  }

  // Return a tuple including the updated results
  return std::make_tuple(poles, residues, polys, fit, rmserr);
}


//
// Python Module and Docstrings
//
namespace py = pybind11;

PYBIND11_MODULE(vectfit, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Vector Fitting Algorithm
        -------------------------
        .. currentmodule:: vectfit
        .. autosummary::
           :toctree: _generate

           vectfit
           evaluate
    )pbdoc";

    m.def("vectfit", &vectfit, R"pbdoc(
        Fast Relaxed Vector Fitting function

        A robust numerical method for rational approximation. It updates the
        poles and calculates residues based on guessed poles.

        Parameters
        ----------
        f : numpy.ndarray
            A 2D array of the sample signals to be fitted, (Nv, Ns)
        s : numpy.ndarray
            A 1D array of the sample points, (Ns)
        poles : numpy.ndarray [complex]
            Initial poles, real or complex conjugate pairs, (N)
        weight : numpy.ndarray
            2D array for weighting f, to control the accuracy of the
            approximation, (Nv, Ns)
        n_polys : int
            Number of polynomial coefficients to be fitted, [0, 11]
        skip_pole : bool
            Whether or not to skip the calculation of poles
        skip_res : bool
            Whether or not to skip the calculation of residues (including the
            polynomials)

        Returns
        -------
        Tuple : (numpy.ndarray [complex], numpy.ndarray [complex], numpy.ndarray, numpy.ndarray, float)
            The updated poles, residues, polynomial coefficients,
            fitted signals on the sample points, root mean square error

    )pbdoc", py::arg("f"), py::arg("s"), py::arg("poles"), py::arg("weight"),
    py::arg("n_polys") = 0, py::arg("skip_pole") = false,
    py::arg("skip_res") = false);

    m.def("evaluate", &evaluate, R"pbdoc(
        Multipole formalism evaluation function

        Evaluate the multipole formalism at given points with known poles,
        residues and polynomial coefficients.

        Parameters
        ----------
        s : numpy.ndarray
            A 1D array of the points, (Ns)
        poles : numpy.ndarray [complex]
            A 1D array of the poles, (N)
        residues : numpy.ndarray [complex]
            2D array of residues, (Nv, N)
        polys : numpy.ndarray
            Polynomial coefficients (0-th to Nc-th order), (Nv, Nc)

        Returns
        -------
        f : numpy.ndarray
            the result array of multipole formalism (real part)

    )pbdoc", py::arg("s"), py::arg("poles"), py::arg("residues"),
    py::arg("polys") = (xt::pyarray<double>) {});

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "0.1";
#endif
}
