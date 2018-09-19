#ifndef VECTFIT_H
#define VECTFIT_H

#include <complex>
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp" // Numpy bindings

//! Fast Relaxed Vector Fitting function
extern std::tuple<xt::pyarray<std::complex<double>>, xt::pyarray<double>,
           xt::pyarray<std::complex<double>>, double, xt::pyarray<double>> 
vectfit(xt::pyarray<double> f, xt::pyarray<double> s,
             xt::pyarray<std::complex<double>> poles, xt::pyarray<double> weight,
             int n_polys, bool skip_pole = false, bool skip_res = false);

#endif // VECTFIT_H
