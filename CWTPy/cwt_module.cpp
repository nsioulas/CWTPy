
// CWTPy/cwt_module.cpp
// CWTPy: A fast, cross‑platform continuous wavelet transform (CWT) library
// using FFTW and OpenMP.
// This module computes the CWT using an L2‐normalized Morlet wavelet.
// It returns the coefficients, scales, frequencies, and a PSD normalization factor.
// The PSD normalization factor is computed as:
//    psd_norm = (4π dt) / (C ω₀ T),
// where T = N*dt and C is the admissibility constant, defined by
//    C = (1/√π) ∫₀∞ exp[–(ω – ω₀)²] / ω dω.
// For ω₀=6, typical C ≈ 0.776 (may vary slightly with integration parameters).
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    // Full Python API must be included first!
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------------------------------------------------
// Morlet normalization factor: ψ(t) = π^(–1/4) exp(i ω₀ t) exp(–t²/2)
// so that ∫|ψ(t)|² dt = 1.
// ---------------------------------------------------------------------
static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}

// ---------------------------------------------------------------------
// Compute the admissibility constant C for the Morlet wavelet.
// We use Simpson's rule to approximate:
//    C = (1/√π) ∫₀∞ exp[–(ω – ω₀)²] / ω dω
// For ω₀ = 6, this should be close to 0.776.
// ---------------------------------------------------------------------
static double compute_admissibility(double omega0) {
    const double eps = 1e-6;
    double upper = omega0 + 10.0;  // upper limit where the integrand is negligible.
    const int nsteps = 100000;     // even number for Simpson's rule.
    double h = (upper - eps) / nsteps;
    double sum = 0.0;
    for (int i = 0; i <= nsteps; i++) {
        double omega = eps + i * h;
        double f = std::exp(-std::pow(omega - omega0, 2)) / omega;
        if (i == 0 || i == nsteps)
            sum += f;
        else if (i % 2 == 1)
            sum += 4 * f;
        else
            sum += 2 * f;
    }
    double integral = (h / 3.0) * sum;
    return integral / std::sqrt(M_PI);
}

// ---------------------------------------------------------------------
// Convert frequency to scale using the relation:
//    freq = (omega0/(2π s)) * (1 + 1/(2ω0²))
// So, s = (omega0/(2π freq)) * (1 + 1/(2ω0²))
// ---------------------------------------------------------------------
static inline double freq_to_scale(double freq, double omega0) {
    if (freq <= 0.0)
        throw std::runtime_error("freq_to_scale: frequency must be > 0");
    double corr = 1.0 + 1.0/(2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * freq)) * corr;
}

// ---------------------------------------------------------------------
// Convert scale to frequency (inverse relation).
// ---------------------------------------------------------------------
static inline double scale_to_freq(double s, double omega0) {
    double corr = 1.0 + 1.0/(2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * s)) * corr;
}

// ---------------------------------------------------------------------
// Build log-spaced scales from s0 to s1 with multiplier a = 2^(1/nv) per octave.
// ---------------------------------------------------------------------
static std::vector<double> make_scales_log(double s0, double s1, int nv) {
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
   ----------------
   Computes the continuous wavelet transform (CWT) of a real 1D signal
   using an L2-normalized Morlet wavelet.

   Parameters:
       signal    : 1D numpy array (real input signal)
       dt        : Sampling interval (seconds)
       nv        : Voices per octave
       omega0    : Morlet wavelet parameter (e.g., 6.0)
       min_freq  : Minimum frequency (Hz); if <=0, defaults to 1/(N*dt)
       max_freq  : Maximum frequency (Hz); if <=0, defaults to fs/2
       use_omp   : If true, parallelize over scales using OpenMP
       norm_mult : Multiplier applied to the normalization constant (default 1.0)
                  Use this to fine-tune the overall amplitude for PSD comparisons.

   Returns:
       A tuple (W, scales, freqs, psd_norm) where:
         - W       : 2D numpy array of shape (num_scales, N) with CWT coefficients.
         - scales  : 1D array of scales.
         - freqs   : 1D array of frequencies corresponding to each scale.
         - psd_norm: Normalization factor for converting wavelet power to a PSD,
                     computed as (4π dt)/(C ω0 T) where T=N*dt and C is the admissibility constant.
*/
py::tuple cwt_morlet_full(
    py::array_t<double> signal,
    double dt,
    int nv,
    double omega0,
    double min_freq,
    double max_freq,
    bool use_omp,
    double norm_mult = 1.0
) {
    // 1) Parse input and define frequency bounds.
    auto buf = signal.request();
    if (buf.ndim != 1)
        throw std::runtime_error("signal must be 1D");
    int N = buf.shape[0];
    if (N < 2)
        throw std::runtime_error("Signal too short.");
    const double* sig_ptr = static_cast<const double*>(buf.ptr);
    double fs = 1.0 / dt;
    if (max_freq <= 0.0)
        max_freq = fs / 2.0;
    if (min_freq <= 0.0)
        min_freq = 1.0 / (N * dt);
    if (min_freq >= max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range.");

    // Convert frequency bounds to scales.
    double smin = freq_to_scale(max_freq, omega0); // smallest scale.
    double smax = freq_to_scale(min_freq, omega0); // largest scale.
    if (smin >= smax)
        throw std::runtime_error("Scale range is invalid. Check frequency bounds.");
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

    // 3) Prepare output container.
    std::vector<std::complex<double>> W_data(num_scales * N);
    std::vector<double> omega_vec(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++) {
        double f_k = (k <= N / 2) ? k * df : -(N - k) * df;
        omega_vec[k] = 2.0 * M_PI * f_k;
    }
    double norm = morlet_factor() * norm_mult;

    // 4) Compute inverse FFT for each scale.
#ifdef _OPENMP
    if (use_omp) {
        if (!fftw_init_threads()) {
            throw std::runtime_error("Failed to initialize FFTW threads.");
        }
        #pragma omp parallel
        {
            fftw_plan_with_nthreads(1);
            fftw_complex* freq_prod_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_complex* inv_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_plan plan_bwd_local = fftw_plan_dft_1d(N, freq_prod_local, inv_local, FFTW_BACKWARD, FFTW_ESTIMATE);
            #pragma omp for schedule(static)
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
        }
    } else
#endif
    {
        fftw_complex* freq_prod = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        fftw_complex* inv = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
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

    // 5) Build frequency array from scales.
    std::vector<double> freq_arr(num_scales);
    for (int i = 0; i < num_scales; i++) {
        freq_arr[i] = scale_to_freq(scales[i], omega0);
    }

    // 6) Compute the PSD normalization factor.
    double T = N * dt;
    double C_val = compute_admissibility(omega0);
    double psd_norm = (4.0 * M_PI * dt) / (C_val * omega0 * T);

    // 7) Convert results to Python arrays.
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
        double* s_ptr = (double*) s_buf.ptr;
        double* f_ptr = (double*) f_buf.ptr;
        for (int i = 0; i < num_scales; i++) {
            s_ptr[i] = scales[i];
            f_ptr[i] = freq_arr[i];
        }
    }

    // Return (W, scales, freqs, psd_norm)
    return py::make_tuple(W_py, scales_py, freqs_py, psd_norm);
}

PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "CWTPy: A fast Morlet CWT using FFTW with self-consistent normalization and optional OpenMP.";
    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv") = 32,
          py::arg("omega0") = 6.0,
          py::arg("min_freq") = 0.0,
          py::arg("max_freq") = 0.0,
          py::arg("use_omp") = false,
          py::arg("norm_mult") = 1.0,
R"doc(
Compute the Morlet continuous wavelet transform (CWT) of a real 1D signal.

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
norm_mult : float, optional
    A multiplier applied to the normalization constant (default=1.0).

Returns
-------
tuple
    (W, scales, freqs, psd_norm) where:
      - W      : np.ndarray of shape (num_scales, N) with CWT coefficients.
      - scales : 1D array of scales.
      - freqs  : 1D array of frequencies corresponding to each scale.
      - psd_norm : Normalization factor for converting wavelet power to a PSD:
                   psd_norm = (4π dt) / (C ω0 T), where T = N*dt and C is the admissibility constant.
)doc");
}
