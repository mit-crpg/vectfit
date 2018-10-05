#ifndef VECTFIT_H
#define VECTFIT_H

#include <complex>
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp" // Numpy bindings

//! Fast Relaxed Vector Fitting function
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
        bool skip_res = false);

//! Multipole formalism evaluation function
xt::pyarray<double>
evaluate(xt::pyarray<double> s,
         xt::pyarray<std::complex<double>> poles,
         xt::pyarray<std::complex<double>> residues,
         xt::pyarray<double> polys = (xt::pyarray<double>) {});

#endif // VECTFIT_H
