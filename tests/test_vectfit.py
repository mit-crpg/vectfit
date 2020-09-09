import vectfit as m
from unittest import TestCase
import numpy as np


class VectfitTest(TestCase):

    def test_vector(self):
        """Test vectfit with vector samples, and simple poles.
           It is expected to get exact results with one iteration.
        """
        Ns = 101
        test_s = np.linspace(3., 7., Ns)
        test_poles = [5.0+0.1j, 5.0-0.1j]
        test_residues = [[0.5-11.0j, 0.5+11.0j],
                         [1.5-20.0j, 1.5+20.0j]]
        f = np.zeros([2, Ns])
        f[0, :] = np.real(test_residues[0][0]/(test_s - test_poles[0]) + \
                  test_residues[0][1]/(test_s - test_poles[1]))
        f[1, :] = np.real(test_residues[1][0]/(test_s - test_poles[0]) + \
                  test_residues[1][1]/(test_s - test_poles[1]))
        weight = 1.0/f
        init_poles = [3.5 + 0.035j, 3.5 - 0.035j]
        poles, residues, cf, fit, rms = m.vectfit(f, test_s, init_poles, weight)
        np.testing.assert_allclose(test_poles, poles, rtol=1e-7)
        np.testing.assert_allclose(test_residues, residues, rtol=1e-7)
        np.testing.assert_allclose(f, fit, rtol=1e-5)

    def test_poly(self):
        """Test vectfit with polynomials."""
        # contructing signal
        Ns = 201
        test_s = np.linspace(0., 5., Ns)
        test_poles = [-20.0+30.0j, -20.0-30.0j]
        test_residues = [[5.0+10.0j, 5.0-10.0j]]
        test_polys = [[1.0, 2.0, 0.3]]
        f = m.evaluate(test_s, test_poles, test_residues, test_polys)
        weight = 1.0/f
        # intial poles
        init_poles = [2.5 + 0.025j, 2.5 - 0.025j]
        # iteration 1
        poles, residues, cf, fit, rms = m.vectfit(f, test_s, init_poles,
                                                  weight, n_polys=3)
        # iteration 2
        poles, residues, cf, fit, rms = m.vectfit(f, test_s, poles,
                                                  weight, n_polys=3)
        np.testing.assert_allclose(test_poles, poles, rtol=1e-5)
        np.testing.assert_allclose(test_residues, residues, rtol=1e-5)
        np.testing.assert_allclose(test_polys, cf, rtol=1e-5)
        np.testing.assert_allclose(f, fit, rtol=1e-4)

    def test_real_poles(self):
        """Test vectfit with more poles including real poles"""
        # contructing signal
        Ns = 5000
        test_s = np.linspace(1.0e-2, 5.e3, Ns)
        test_poles = np.array([
             9.709261771920490e+02 + 0.0j,
            -1.120960794075339e+03 + 0.0j,
             1.923889557426567e+00 + 7.543700246109742e+01j,
             1.923889557426567e+00 - 7.543700246109742e+01j,
             1.159741300380281e+02 + 3.595650922556496e-02j,
             1.159741300380281e+02 - 3.595650922556496e-02j,
             1.546932165729394e+02 + 8.728391144940301e-02j,
             1.546932165729394e+02 - 8.728391144940301e-02j,
             2.280349190818197e+02 + 2.814037559718684e-01j,
             2.280349190818197e+02 - 2.814037559718684e-01j,
             2.313004772627853e+02 + 3.004628477692201e-01j,
             2.313004772627853e+02 - 3.004628477692201e-01j,
             2.787470098364861e+02 + 3.414179169920170e-01j,
             2.787470098364861e+02 - 3.414179169920170e-01j,
             3.570711338764254e+02 + 4.485587371149193e-01j,
             3.570711338764254e+02 - 4.485587371149193e-01j,
             4.701059001346060e+02 + 6.598089307174224e-01j,
             4.701059001346060e+02 - 6.598089307174224e-01j,
             7.275819506342254e+02 + 1.189678974845038e+03j,
             7.275819506342254e+02 - 1.189678974845038e+03j
        ])
        test_residues = np.array([[
            -3.269879776751686e+07 + 0.0j,
             1.131087935798761e+09 + 0.0j,
             1.634151281869857e+04 + 2.251103589277891e+05j,
             1.634151281869857e+04 - 2.251103589277891e+05j,
             3.281792303833561e+03 - 1.756079516325274e+04j,
             3.281792303833561e+03 + 1.756079516325274e+04j,
             1.110800880243503e+04 - 4.324813594540043e+04j,
             1.110800880243503e+04 + 4.324813594540043e+04j,
             8.812700704117636e+04 - 2.256520243571103e+05j,
             8.812700704117636e+04 + 2.256520243571103e+05j,
             5.842090495551535e+04 - 1.442159380741478e+05j,
             5.842090495551535e+04 + 1.442159380741478e+05j,
             1.339410514130921e+05 - 2.640767909713812e+05j,
             1.339410514130921e+05 + 2.640767909713812e+05j,
             2.211245633333130e+05 - 3.222447758311512e+05j,
             2.211245633333130e+05 + 3.222447758311512e+05j,
             4.124430059785149e+05 - 4.076023108323907e+05j,
             4.124430059785149e+05 + 4.076023108323907e+05j,
             1.607378314999252e+09 - 1.401163320110452e+08j,
             1.607378314999252e+09 + 1.401163320110452e+08j
        ]])
        f = np.zeros((1, Ns))
        for p, r in zip(test_poles, test_residues[0]):
            f[0] += (r/(test_s - p)).real
        weight = 1.0/f
        # intial poles
        poles = np.linspace(1.1e-2, 4.8e+3, 10);
        poles = poles + poles*0.01j
        poles = np.sort(np.append(poles, np.conj(poles)))
        # VF iterations
        for i in range(10):
            poles, residues, cf, fit, rms = m.vectfit(f, test_s, poles, weight)
        np.testing.assert_allclose(np.sort(test_poles), np.sort(poles))
        np.testing.assert_allclose(np.sort(test_residues), np.sort(residues))
        np.testing.assert_allclose(f, fit, 1e-3)

    def test_large(self):
        """Test vectfit with a large set of poles and samples"""
        Ns = 20000
        N = 1000
        s = np.linspace(1.0e-2, 5.e3, Ns)
        poles = np.linspace(1.1e-2, 4.8e+3, N//2);
        poles = poles + poles*0.01j
        poles = np.sort(np.append(poles, np.conj(poles)))
        residues = np.linspace(1e+2, 1e+6, N//2);
        residues = residues + residues*0.5j
        residues = np.sort(np.append(residues, np.conj(residues)))
        residues = residues.reshape((1, N))
        f = np.zeros((1, Ns))
        for p, r in zip(poles, residues[0]):
            f[0] += (r/(s - p)).real
        weight = 1.0/f
        poles_init = np.linspace(1.2e-2, 4.7e+3, N//2);
        poles_init = poles_init + poles_init*0.01j
        poles_init = np.sort(np.append(poles_init, np.conj(poles_init)))

        poles_fit, residues_fit, cf, f_fit, rms = m.vectfit(f, s, poles_init, weight)

        np.testing.assert_allclose(f, f_fit, 1e-3)

    def test_evaluate(self):
        """Test evaluate function"""
        Ns = 101
        s = np.linspace(-5., 5., Ns)
        poles = [-2.0+30.0j, -2.0-30.0j]
        residues = [5.0+10.0j, 5.0-10.0j]
        f_ref = np.zeros([1, Ns])
        f_ref[0, :] = np.real(residues[0]/(s - poles[0]) + \
                              residues[1]/(s - poles[1]))
        f = m.evaluate(s, poles, residues)
        np.testing.assert_allclose(f_ref, f)

        polys = [1.0, 2.0, 0.3]
        for n, c in enumerate(polys):
            f_ref[0, :] += c*np.power(s, n)
        f = m.evaluate(s, poles, residues, polys)
        np.testing.assert_allclose(f_ref, f)

        poles = [5.0+0.1j, 5.0-0.1j]
        residues = [[0.5-11.0j, 0.5+11.0j],
                    [1.5-20.0j, 1.5+20.0j]]
        polys = [[1.0, 2.0, 0.3], [4.0, -2.0, -10.0]]
        f_ref = np.zeros([2, Ns])
        f_ref[0, :] = np.real(residues[0][0]/(s - poles[0]) + \
                              residues[0][1]/(s - poles[1]))
        f_ref[1, :] = np.real(residues[1][0]/(s - poles[0]) + \
                              residues[1][1]/(s - poles[1]))
        for n, c in enumerate(polys[0]):
            f_ref[0, :] += c*np.power(s, n)
        for n, c in enumerate(polys[1]):
            f_ref[1, :] += c*np.power(s, n)
        f = m.evaluate(s, poles, residues, polys)
        np.testing.assert_allclose(f_ref, f)
