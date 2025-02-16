// cwt_module.cpp
// CWTPy: A fast, cross‑platform CWT implementation using FFTW and OpenMP.
// This module computes the continuous wavelet transform using an L2‑normalized
// Morlet wavelet and returns the coefficients, scales, and frequencies.
// 
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    // Include full Python API first!

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

// L2 normalization factor for Morlet (pi^(-1/4)).
static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}

// eqn(12): freq = (omega0/(2π s)) * (1 + 1/(2ω0^2)).
// For now, we set correction factor to 1.0 (you can adjust as needed).
static inline double scale_to_freq(double s, double omega0) {
    double corr = 1.0;  // adjust if needed: 1.0 + 1.0/(2.0*omega0*omega0)
    return (omega0 / (2.0 * M_PI * s)) * corr;
}

// Inverted eqn(12): s = (omega0/(2π freq)) * (1 + 1/(2ω0^2)).
static inline double freq_to_scale(double freq, double omega0) {
    if (freq <= 0.0)
        throw std::runtime_error("freq_to_scale: frequency must be > 0");
    double corr = 1.0;  // adjust if needed
    return (omega0 / (2.0 * M_PI * freq)) * corr;
}

// Build log-spaced scales from s0 -> s1 with factor a = 2^(1/nv) per octave.
std::vector<double> make_scales_log(double s0, double s1, int nv) {
    if (s0 <= 0 || s1 <= s0)
        throw std::runtime_error("make_scales_log: invalid scale range");
    double a = std::pow(2.0, 1.0 / double(nv));
    std::vector<double> scales;
    for (double s = s0; s <= s1; s *= a) {
        scales.push_back(s);
    }
    return scales;
}

/*
   cwt_morlet_full:
   ---------------
   signal   : 1D real array.
   dt       : sampling interval.
   nv       : voices per octave.
   omega0   : Morlet wavelet parameter.
   min_freq, max_freq:
              if <= 0, use defaults: [1/(N*dt), fs/2].
   use_omp  : if true, parallelize over scales using OpenMP.
   Returns a tuple (W, scales, freqs).
*/
py::tuple cwt_morlet_full(
    py::array_t<double> signal,
    double dt,
    int nv,
    double omega0,
    double min_freq,
    double max_freq,
    bool use_omp
) {
    // 1) Parse input and define frequency range.
    auto buf = signal.request();
    if (buf.ndim != 1)
        throw std::runtime_error("signal must be 1D");
    int N = buf.shape[0];
    if (N < 2)
        throw std::runtime_error("Signal too short.");
    const double* sig_ptr = buf.ptr;
    double fs = 1.0 / dt;
    if (max_freq <= 0.0)
        max_freq = fs / 2.0;      // default to Nyquist.
    if (min_freq <= 0.0)
        min_freq = 1.0 / (N * dt); // default to lowest resolvable.
    if (min_freq >= max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range.");

    // Convert frequency bounds to scales.
    double smin = freq_to_scale(max_freq, omega0); // smallest scale.
    double smax = freq_to_scale(min_freq, omega0); // largest scale.
    if (smin >= smax)
        throw std::runtime_error("Scale range is invalid. Check freq bounds.");
    std::vector<double> scales = make_scales_log(smin, smax, nv);
    int num_scales = static_cast<int>(scales.size());

    // 2) Compute forward FFT of the signal.
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    for (int i = 0; i < N; i++) {
        in[i][0] = sig_ptr[i];
        in[i][1] = 0.0;
    }
    fftw_plan plan_fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_fwd);
    fftw_destroy_plan(plan_fwd);
    fftw_free(in);
    std::vector<std::complex<double>> sig_fft(N);
    for (int i = 0; i < N; i++) {
        sig_fft[i] = { out[i][0], out[i][1] };
    }
    fftw_free(out);

    // 3) Prepare output container for coefficients.
    std::vector<std::complex<double>> W_data(num_scales * N);
    // Build an angular frequency array.
    std::vector<double> omega_vec(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++) {
        double f_k = (k <= N/2) ? k * df : -(N - k) * df;
        omega_vec[k] = 2.0 * M_PI * f_k;
    }
    double norm = morlet_factor(); // normalization constant.

    // 4) Compute inverse FFT for each scale.
    // If OpenMP is enabled, use a parallel loop.
#ifdef _OPENMP
    if (use_omp) {
        // Initialize FFTW threading once.
        fftw_init_threads();
        #pragma omp parallel
        {
            // Each thread uses one thread for its own FFTW plan.
            fftw_plan_with_nthreads(1);
            // Allocate thread-local buffers.
            fftw_complex* freq_prod_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_complex* inv_local       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_plan plan_bwd_local = fftw_plan_dft_1d(N, freq_prod_local, inv_local, FFTW_BACKWARD, FFTW_ESTIMATE);

            #pragma omp for schedule(dynamic)
            for (int sidx = 0; sidx < num_scales; sidx++) {
                double s = scales[sidx];
                for (int k = 0; k < N; k++) {
                    double arg = s * omega_vec[k] - omega0;
                    double wavelet = std::exp(-0.5 * arg * arg) * std::sqrt(s) * norm;
                    std::complex<double> val = sig_fft[k] * wavelet;
                    freq_prod_local[k][0] = val.real();
                    freq_prod_local[k][1] = val.imag();
                }
                fftw_execute(plan_bwd_local);
                for (int n = 0; n < N; n++) {
                    double re = inv_local[n][0] / double(N);
                    double im = inv_local[n][1] / double(N);
                    W_data[sidx * N + n] = std::complex<double>(re, im);
                }
            }
            fftw_destroy_plan(plan_bwd_local);
            fftw_free(freq_prod_local);
            fftw_free(inv_local);
        } // end OpenMP region
    } else
#endif
    {
        // Single-threaded version.
        fftw_complex* freq_prod = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* inv       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        fftw_plan plan_bwd = fftw_plan_dft_1d(N, freq_prod, inv, FFTW_BACKWARD, FFTW_ESTIMATE);
        for (int sidx = 0; sidx < num_scales; sidx++) {
            double s = scales[sidx];
            for (int k = 0; k < N; k++) {
                double arg = s * omega_vec[k] - omega0;
                double wavelet = std::exp(-0.5 * arg * arg) * std::sqrt(s) * norm;
                std::complex<double> val = sig_fft[k] * wavelet;
                freq_prod[k][0] = val.real();
                freq_prod[k][1] = val.imag();
            }
            fftw_execute(plan_bwd);
            for (int n = 0; n < N; n++) {
                double re = inv[n][0] / double(N);
                double im = inv[n][1] / double(N);
                W_data[sidx * N + n] = std::complex<double>(re, im);
            }
        }
        fftw_destroy_plan(plan_bwd);
        fftw_free(freq_prod);
        fftw_free(inv);
    }

    // 5) Build frequency array corresponding to scales.
    std::vector<double> freqs(num_scales);
    for (int i = 0; i < num_scales; i++) {
        freqs[i] = scale_to_freq(scales[i], omega0);
    }

    // 6) Convert data to Python objects.
    py::array_t<std::complex<double>> W_py({ num_scales, N });
    {
        auto W_buf = W_py.request();
        auto* W_ptr = (std::complex<double>*)W_buf.ptr;
        std::memcpy(W_ptr, W_data.data(), num_scales * N * sizeof(std::complex<double>));
    }
    py::array_t<double> scales_py(num_scales);
    py::array_t<double> freqs_py(num_scales);
    {
        auto s_buf = scales_py.request();
        auto f_buf = freqs_py.request();
        double* s_ptr = (double*)s_buf.ptr;
        double* f_ptr = (double*)f_buf.ptr;
        for (int i = 0; i < num_scales; i++) {
            s_ptr[i] = scales[i];
            f_ptr[i] = freqs[i];
        }
    }

    return py::make_tuple(W_py, scales_py, freqs_py);
}

PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "Faster Morlet CWT with OpenMP parallelization using per-thread inverse FFT plans";
    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv") = 32,
          py::arg("omega0") = 6.0,
          py::arg("min_freq") = 0.0,
          py::arg("max_freq") = 0.0,
          py::arg("use_omp") = false,
R"doc(
Compute the Morlet continuous wavelet transform of a real 1D signal.

Parameters
----------
signal : np.ndarray (1D)
    Real input signal.
dt : float
    Sampling interval.
nv : int, optional
    Voices per octave (default=32).
omega0 : float, optional
    Morlet wavelet parameter (default=6).
min_freq : float, optional
    Minimum frequency (Hz). If <= 0, defaults to 1/(N*dt).
max_freq : float, optional
    Maximum frequency (Hz). If <= 0, defaults to fs/2.
use_omp : bool, optional
    If True, parallelize the transform over scales using OpenMP.

Returns
-------
tuple
    A tuple (W, scales, freqs) where:
        W      : np.ndarray of shape (num_scales, N) with CWT coefficients.
        scales : 1D array of scales.
        freqs  : 1D array of frequencies corresponding to each scale.
)doc");
}
