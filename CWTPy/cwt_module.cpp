// CWTPy/cwt_module.cpp
// CWTPy: A fast, cross‑platform continuous wavelet transform (CWT) implementation
// using FFTW and optional OpenMP, with a user-specified normalization approach,
// approximate COI, and FFT frequencies returned in Hz.
//
// This module computes the CWT using an L2‑normalized Morlet wavelet and returns:
//    - W:         CWT coefficients (num_scales x N)
//    - scales:    array of scales
//    - freqs:     wavelet frequencies (Hz) mapped from those scales
//    - psd_factor:  the partial PSD factor = (4π)/(C ω₀).  It's up to the user to
//                   further divide by N, T, or #samples used in an averaging, as needed
//                   
//    - fft_freqs:  the FFT frequencies (Hz) for the input signal
//    - coi:        approximate cone of influence (length N), if requested
//

//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    // Must be included first
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
#include <algorithm>

namespace py = pybind11;

//---------------------------------------------------------------------------------
// 1) L2 normalization factor for the Morlet wavelet: ψ(t) = π^(-1/4) e^(i ω₀ t) e^(-t²/2)
//---------------------------------------------------------------------------------
static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}

//---------------------------------------------------------------------------------
// 2) Compute the Morlet admissibility constant C via Simpson's rule
//    C = (1/√π) ∫₀^∞ [exp(-(ω - ω₀)²) / ω ] dω
//    For ω₀=6, typically ~0.776
//---------------------------------------------------------------------------------
static double compute_admissibility(double omega0) {
    const double eps = 1e-6;
    double upper = omega0 + 10.0;  // integration upper limit
    const int nsteps = 100000;     // must be even for Simpson's rule
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

//---------------------------------------------------------------------------------
// 3) freq <-> scale mapping, eqn(12) in your references, including the correction factor
//---------------------------------------------------------------------------------
static inline double freq_to_scale(double freq, double omega0) {
    if (freq <= 0.0)
        throw std::runtime_error("freq_to_scale: frequency must be > 0");
    double corr = 1.0 + 1.0/(2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * freq)) * corr;
}
static inline double scale_to_freq(double s, double omega0) {
    double corr = 1.0 + 1.0/(2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * s)) * corr;
}

//---------------------------------------------------------------------------------
// 4) Build log-spaced scales from s0 to s1 with multiplier a = 2^(1/nv) per octave
//---------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------
// 5) Compute the approximate cone of influence (COI).
//    The user references eqn(13)/(18) in screenshots, but we do a simple approach:
//    COI[t] = dt * sqrt(2) * min(t+1, N-t).
//    You may want to refine or replace this if eqn(13) suggests a different formula
//---------------------------------------------------------------------------------
static std::vector<double> compute_coi(int N, double dt) {
    std::vector<double> coi(N);
    for (int t = 0; t < N; t++) {
        double d = std::min(double(t + 1), double(N - t));
        coi[t] = dt * std::sqrt(2.0) * d;
    }
    return coi;
}

//---------------------------------------------------------------------------------
// 6) Compute the standard FFT frequencies (in Hz) for a signal of length N, sampling freq fs
//---------------------------------------------------------------------------------
static std::vector<double> compute_fft_freqs(int N, double fs) {
    std::vector<double> fft_freqs(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++) {
        if (k <= N/2)
            fft_freqs[k] = k * df;
        else
            fft_freqs[k] = -(N - k) * df;
    }
    return fft_freqs;
}

/*
   cwt_morlet_full:
   ----------------
   Compute the continuous wavelet transform (CWT) of a real 1D signal using an L2-normalized Morlet wavelet.
   Returns a tuple with:
     (W, scales, freqs, psd_factor, fft_freqs[, coi if consider_coi=true])

   The user then can decide how to incorporate sample counts or T in eqn(17) in Python.

   Parameters
   ----------
   signal       : 1D numpy array (real input signal)
   dt           : sampling interval (seconds)
   nv           : voices per octave
   omega0       : Morlet wavelet parameter (e.g. 6.0)
   min_freq     : minimum frequency (Hz); if <=0 => ~1/(N dt)
   max_freq     : maximum frequency (Hz); if <=0 => fs/2
   use_omp      : if true => parallelize over scales using OpenMP
   norm_mult    : multiplier for the wavelet normalization (default=1.0)
   consider_coi : if true => compute & return an approximate COI array
   return_fft   : if true => compute & return the FFT frequency array (in Hz)

   Returns
   -------
   tuple
       - W         : 2D array (num_scales x N) with wavelet coefficients
       - scales    : 1D array of scales
       - freqs     : wavelet frequencies in Hz for each scale
       - psd_factor: partial PSD factor => (4π)/(C ω₀). The user can later
                     multiply by (dt / T) or 1/N or something else as eqn(17) suggests.
       - fft_freqs : 1D array of FFT frequencies (Hz) if return_fft=true, else empty
       - coi       : 1D array of approximate COI if consider_coi=true, else empty
*/
py::tuple cwt_morlet_full(
    py::array_t<double> signal,
    double dt,
    int nv,
    double omega0,
    double min_freq,
    double max_freq,
    bool use_omp,
    double norm_mult,
    bool consider_coi,
    bool return_fft
) {
    // 1) parse input
    auto buf = signal.request();
    if (buf.ndim != 1)
        throw std::runtime_error("signal must be 1D");
    int N = buf.shape[0];
    if (N < 2)
        throw std::runtime_error("Signal too short");
    const double* sig_ptr = static_cast<const double*>(buf.ptr);

    double fs = 1.0 / dt;
    if (max_freq <= 0.0)
        max_freq = fs / 2.0;
    if (min_freq <= 0.0)
        min_freq = 1.0 / (N * dt);
    if (min_freq >= max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range");

    // 2) map freq -> scale
    double smin = freq_to_scale(max_freq, omega0);
    double smax = freq_to_scale(min_freq, omega0);
    if (smin >= smax)
        throw std::runtime_error("Scale range is invalid. Check freq bounds.");
    std::vector<double> scales = make_scales_log(smin, smax, nv);
    int num_scales = static_cast<int>(scales.size());

    // 3) forward FFT
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
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
        sig_fft[i] = std::complex<double>(out[i][0], out[i][1]);
    }
    fftw_free(out);

    // 4) container for wavelet coefficients
    std::vector<std::complex<double>> W_data(num_scales * N);

    // build angular freq array
    std::vector<double> omega_vec(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++) {
        double f_k = (k <= N/2) ? k*df : -(N - k)*df;
        omega_vec[k] = 2.0*M_PI*f_k;
    }
    double norm = morlet_factor() * norm_mult;

    // 5) inverse FFT per scale
#ifdef _OPENMP
    if (use_omp) {
        if (!fftw_init_threads()) {
            throw std::runtime_error("Failed to initialize FFTW threads.");
        }
        #pragma omp parallel
        {
            fftw_plan_with_nthreads(1);
            fftw_complex* freq_prod_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
            fftw_complex* inv_local       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
            fftw_plan plan_bwd_local = fftw_plan_dft_1d(N, freq_prod_local, inv_local, FFTW_BACKWARD, FFTW_ESTIMATE);

            #pragma omp for schedule(static)
            for (int sidx = 0; sidx < num_scales; sidx++) {
                double s = scales[sidx];
                for (int k = 0; k < N; k++) {
                    double arg = s*omega_vec[k] - omega0;
                    double wavelet = std::exp(-0.5*arg*arg) * std::sqrt(s)*norm;
                    std::complex<double> val = sig_fft[k]*wavelet;
                    freq_prod_local[k][0] = val.real();
                    freq_prod_local[k][1] = val.imag();
                }
                fftw_execute(plan_bwd_local);
                for (int n = 0; n < N; n++) {
                    double re = inv_local[n][0]/double(N);
                    double im = inv_local[n][1]/double(N);
                    W_data[sidx*N + n] = std::complex<double>(re, im);
                }
            }
            fftw_destroy_plan(plan_bwd_local);
            fftw_free(freq_prod_local);
            fftw_free(inv_local);
        }
    } else
#endif
    {
        fftw_complex* freq_prod = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        fftw_complex* inv       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        fftw_plan plan_bwd = fftw_plan_dft_1d(N, freq_prod, inv, FFTW_BACKWARD, FFTW_ESTIMATE);
        for (int sidx = 0; sidx < num_scales; sidx++) {
            double s = scales[sidx];
            for (int k = 0; k < N; k++) {
                double arg = s*omega_vec[k] - omega0;
                double wavelet = std::exp(-0.5*arg*arg) * std::sqrt(s)*norm;
                std::complex<double> val = sig_fft[k]*wavelet;
                freq_prod[k][0] = val.real();
                freq_prod[k][1] = val.imag();
            }
            fftw_execute(plan_bwd);
            for (int n = 0; n < N; n++) {
                double re = inv[n][0]/double(N);
                double im = inv[n][1]/double(N);
                W_data[sidx*N + n] = std::complex<double>(re, im);
            }
        }
        fftw_destroy_plan(plan_bwd);
        fftw_free(freq_prod);
        fftw_free(inv);
    }

    // 6) wavelet frequencies in Hz
    std::vector<double> wave_freqs(num_scales);
    for (int i = 0; i < num_scales; i++) {
        wave_freqs[i] = scale_to_freq(scales[i], omega0);
    }

    // 7) partial PSD factor => 4π / (C ω₀). (User can multiply by dt/T or #samples if eqn(17) demands.)
    double C_val = compute_admissibility(omega0);
    double psd_factor = (4.0 * M_PI) / (C_val * omega0);

    // 8) Convert outputs to Python arrays
    py::array_t<std::complex<double>> W_py({num_scales, N});
    {
        auto W_buf = W_py.request();
        auto* W_ptr = (std::complex<double>*) W_buf.ptr;
        std::memcpy(W_ptr, W_data.data(), num_scales*N*sizeof(std::complex<double>));
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
            f_ptr[i] = wave_freqs[i];
        }
    }

    // Always in Hz
    py::array_t<double> fft_freqs_py(0); // empty by default
    if (return_fft) {
        std::vector<double> fft_freqs = compute_fft_freqs(N, fs);
        fft_freqs_py.resize({(size_t)N});
        auto fft_buf = fft_freqs_py.request();
        double* ptr = (double*) fft_buf.ptr;
        for (int i = 0; i < N; i++) {
            ptr[i] = fft_freqs[i];
        }
    }

    // 9) If user wants COI
    py::array_t<double> coi_py(0);
    if (consider_coi) {
        std::vector<double> coi = compute_coi(N, dt);
        coi_py.resize({(size_t)N});
        auto coi_buf = coi_py.request();
        double* cptr = (double*) coi_buf.ptr;
        for (int i = 0; i < N; i++) {
            cptr[i] = coi[i];
        }
    }

    // Build return
    // Return: (W, scales, freqs, psd_factor, fft_freqs, [coi if consider_coi])
    if (consider_coi) {
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py, coi_py);
    } else {
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py);
    }
}

PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "CWTPy: final revision with partial PSD factor, wavelet freq in Hz, FFT freq in Hz, optional COI, per eqn(17) references.";
    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv") = 32,
          py::arg("omega0") = 6.0,
          py::arg("min_freq") = 0.0,
          py::arg("max_freq") = 0.0,
          py::arg("use_omp") = false,
          py::arg("norm_mult") = 1.0,
          py::arg("consider_coi") = false,
          py::arg("return_fft") = false,
R"doc(
Compute the Morlet continuous wavelet transform (CWT) of a real 1D signal.
Now returning partial PSD factor = 4π/(C ω₀), wavelet frequencies in Hz, FFT frequencies in Hz,
and an approximate cone of influence if requested.

Parameters
----------
signal : 1D numpy array
dt : float
nv : int
omega0 : float
min_freq : float
max_freq : float
use_omp : bool
norm_mult : float
consider_coi : bool
return_fft : bool

Returns
-------
- If consider_coi=False: (W, scales, wavelet_freqs, psd_factor, fft_freqs)
- If consider_coi=True : (W, scales, wavelet_freqs, psd_factor, fft_freqs, coi)

where psd_factor = 4π / (C ω₀). The user can multiply by dt/(N dt) or 1/N if eqn(17) or eqn(13)
requires it, or incorporate sample counts in the final PSD step.
)doc");
}
