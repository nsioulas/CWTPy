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

// CWTPy/cwt_module.cpp
// CWTPy: A single unified C++ module for:
//   1) cwt_morlet_full  : Morlet wavelet CWT
//   2) local_gaussian_mean : local Gaussian mean (Equation (22)) with normalization
//
// Both are exposed via Pybind11 in the same library.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)


// CWTPy/cwt_module.cpp
// CWTPy: A fast, cross‑platform continuous wavelet transform (CWT)
// and local Gaussian mean library using FFTW and optional OpenMP,
// with a user-specified normalization approach, approximate COI, and 
// FFT frequencies returned in Hz.
//
// This module computes the CWT using an L2‑normalized Morlet wavelet and returns:
//    - W:         CWT coefficients (num_scales x N)
//    - scales:    array of scales
//    - freqs:     wavelet frequencies (Hz) mapped from those scales
//    - psd_factor: partial PSD factor = 4π/(C ω₀)
//    - fft_freqs: FFT frequencies (Hz) for the input signal
//    - coi:       approximate cone of influence (if requested)
//
// It also implements a local Gaussian mean (eqn(22)):
//    B_n(s) = ( Σ_m B_m exp(- (t_n-t_m)²/(2*lam²*s²)) ) / ( Σ_m exp(- (t_n-t_m)²/(2*lam²*s²)) )
//
// For efficiency, the local Gaussian mean function restricts m to indices
// with |t_n - t_m| <= 3 * lam * s using binary search.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

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

// CWTPy/cwt_module.cpp
// CWTPy: A single unified C++ module for:
//   1) cwt_morlet_full  : Morlet wavelet CWT
//   2) local_gaussian_mean : local Gaussian mean (Equation (22)) with normalization
//
// Both are exposed via Pybind11 in the same library.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)


// CWTPy/cwt_module.cpp
// CWTPy: A fast, cross‑platform continuous wavelet transform (CWT)
// and local Gaussian mean library using FFTW and optional OpenMP,
// with a user-specified normalization approach, approximate COI, and 
// FFT frequencies returned in Hz.
//
// This module computes the CWT using an L2‑normalized Morlet wavelet and returns:
//    - W:         CWT coefficients (num_scales x N)
//    - scales:    array of scales
//    - freqs:     wavelet frequencies (Hz) mapped from those scales
//    - psd_factor: partial PSD factor = 4π/(C ω₀)
//    - fft_freqs: FFT frequencies (Hz) for the input signal
//    - coi:       approximate cone of influence (if requested)
//
// It also implements a local Gaussian mean (eqn(22)):
//    B_n(s) = ( Σ_m B_m exp(- (t_n-t_m)²/(2*lam²*s²)) ) / ( Σ_m exp(- (t_n-t_m)²/(2*lam²*s²)) )
//
// For efficiency, the local Gaussian mean function restricts m to indices
// with |t_n - t_m| <= 3 * lam * s using binary search.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

// CWTPy/cwt_module.cpp
// CWTPy: A single unified C++ module for:
//   1) cwt_morlet_full  : Morlet wavelet CWT
//   2) local_gaussian_mean : local Gaussian mean (Equation (22)) with normalization
//
// We remove calls to fftw_init_threads() or fftw_plan_with_nthreads() to avoid crashes on macOS.
// Instead, we parallelize the outer loop over scales with OpenMP, but each FFT is single-threaded.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>  // Must be included first!
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

//------------------------------------------------------------------------------
// 1) Morlet wavelet helpers
//------------------------------------------------------------------------------
static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}

static double compute_admissibility(double omega0) {
    const double eps = 1e-6;
    double upper = omega0 + 10.0;
    const int nsteps = 100000;  // must be even for Simpson's rule
    double h = (upper - eps) / nsteps;
    double sum = 0.0;
    for (int i = 0; i <= nsteps; i++) {
        double omega = eps + i * h;
        double f = std::exp(-std::pow(omega - omega0, 2)) / omega;
        if (i == 0 || i == nsteps) {
            sum += f;
        } else if (i % 2 == 1) {
            sum += 4.0 * f;
        } else {
            sum += 2.0 * f;
        }
    }
    double integral = (h / 3.0) * sum;
    return integral / std::sqrt(M_PI);
}

static inline double freq_to_scale(double freq, double omega0) {
    if (freq <= 0.0) {
        throw std::runtime_error("freq_to_scale: frequency must be > 0");
    }
    double corr = 1.0 + 1.0 / (2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * freq)) * corr;
}

static inline double scale_to_freq(double s, double omega0) {
    double corr = 1.0 + 1.0 / (2.0 * omega0 * omega0);
    return (omega0 / (2.0 * M_PI * s)) * corr;
}

static std::vector<double> make_scales_log(double s0, double s1, int nv) {
    if (s0 <= 0 || s1 <= s0) {
        throw std::runtime_error("make_scales_log: invalid scale range");
    }
    double a = std::pow(2.0, 1.0 / double(nv));
    std::vector<double> scales;
    for (double s = s0; s <= s1; s *= a) {
        scales.push_back(s);
    }
    return scales;
}

static std::vector<double> compute_coi(int N, double dt) {
    std::vector<double> coi(N);
    for (int t = 0; t < N; t++) {
        double d = std::min(double(t + 1), double(N - t));
        coi[t] = dt * std::sqrt(2.0) * d;
    }
    return coi;
}

static std::vector<double> compute_fft_freqs(int N, double fs) {
    std::vector<double> fft_freqs(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++) {
        if (k <= N / 2) {
            fft_freqs[k] = k * df;
        } else {
            fft_freqs[k] = -(N - k) * df;
        }
    }
    return fft_freqs;
}

//------------------------------------------------------------------------------
// 2) cwt_morlet_full
//------------------------------------------------------------------------------
/*
   cwt_morlet_full:
   Computes the Morlet CWT of a 1D real signal.
   Returns a tuple:
     (W, scales, wavelet_freqs, psd_factor, fft_freqs[, coi]).
   psd_factor = 4π/(C ω₀), where C is computed via Simpson's rule.
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
    if (buf.ndim != 1) {
        throw std::runtime_error("signal must be 1D");
    }
    int N = buf.shape[0];
    if (N < 2) {
        throw std::runtime_error("Signal too short");
    }
    const double* sig_ptr = static_cast<const double*>(buf.ptr);

    double fs = 1.0 / dt;
    if (max_freq <= 0.0) {
        max_freq = fs / 2.0;
    }
    if (min_freq <= 0.0) {
        min_freq = 1.0 / (N * dt);
    }
    if (min_freq >= max_freq) {
        throw std::runtime_error("min_freq >= max_freq => invalid range");
    }

    // freq -> scale
    double smin = freq_to_scale(max_freq, omega0);
    double smax = freq_to_scale(min_freq, omega0);
    if (smin >= smax) {
        throw std::runtime_error("Scale range is invalid. Check freq bounds.");
    }
    std::vector<double> scales = make_scales_log(smin, smax, nv);
    int num_scales = static_cast<int>(scales.size());

    // 2) forward FFT
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
        sig_fft[i] = std::complex<double>(out[i][0], out[i][1]);
    }
    fftw_free(out);

    // 3) container for wavelet coefficients
    std::vector<std::complex<double>> W_data(num_scales * N);

    // 4) build angular freq array
    std::vector<double> omega_vec(N);
    double df_val = fs / double(N);
    for (int k = 0; k < N; k++) {
        double f_k = (k <= N / 2) ? k * df_val : -(N - k) * df_val;
        omega_vec[k] = 2.0 * M_PI * f_k;
    }
    double norm = morlet_factor() * norm_mult;

    // 5) inverse FFT per scale, parallelize outer loop with OpenMP, but single-thread each FFT
#ifdef _OPENMP
    if (use_omp) {
        // We do NOT call fftw_init_threads() or fftw_plan_with_nthreads().
        // We simply parallelize over scales with a single-thread plan for each scale.
        #pragma omp parallel for schedule(static)
        for (int sidx = 0; sidx < num_scales; sidx++) {
            // allocate local buffers
            fftw_complex* freq_prod_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_complex* inv_local       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

            // create a single-thread plan
            fftw_plan plan_bwd_local = fftw_plan_dft_1d(N,
                                                        freq_prod_local,
                                                        inv_local,
                                                        FFTW_BACKWARD,
                                                        FFTW_ESTIMATE);

            double s = scales[sidx];
            for (int k = 0; k < N; k++) {
                double arg = s * omega_vec[k] - omega0;
                double wavelet = std::exp(-0.5 * arg * arg) * std::sqrt(s) * norm;
                std::complex<double> val = sig_fft[k] * wavelet;
                freq_prod_local[k][0] = val.real();
                freq_prod_local[k][1] = val.imag();
            }

            // execute iFFT
            fftw_execute(plan_bwd_local);

            // store results
            for (int n = 0; n < N; n++) {
                double re = inv_local[n][0] / double(N);
                double im = inv_local[n][1] / double(N);
                W_data[sidx * N + n] = std::complex<double>(re, im);
            }

            // cleanup
            fftw_destroy_plan(plan_bwd_local);
            fftw_free(freq_prod_local);
            fftw_free(inv_local);
        }
    } else
#endif
    {
        // Non-OMP version, single-threaded
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

    // 6) wavelet frequencies in Hz
    std::vector<double> wave_freqs(num_scales);
    for (int i = 0; i < num_scales; i++){
        wave_freqs[i] = scale_to_freq(scales[i], omega0);
    }

    // 7) partial PSD factor => 4π/(C ω₀)
    double C_val = compute_admissibility(omega0);
    double psd_factor = (4.0 * M_PI) / (C_val * omega0);

    // 8) Convert outputs to Python arrays
    py::array_t<std::complex<double>> W_py({num_scales, N});
    {
        auto W_buf = W_py.request();
        auto* W_ptr = (std::complex<double>*) W_buf.ptr;
        std::memcpy(W_ptr, W_data.data(), num_scales * N * sizeof(std::complex<double>));
    }
    py::array_t<double> scales_py(num_scales);
    py::array_t<double> freqs_py(num_scales);
    {
        auto s_buf = scales_py.request();
        auto f_buf = freqs_py.request();
        double* s_ptr = (double*) s_buf.ptr;
        double* f_ptr = (double*) f_buf.ptr;
        for (int i = 0; i < num_scales; i++){
            s_ptr[i] = scales[i];
            f_ptr[i] = wave_freqs[i];
        }
    }

    // optional FFT frequencies
    py::array_t<double> fft_freqs_py(0);
    if (return_fft) {
        std::vector<double> fft_freqs = compute_fft_freqs(N, fs);
        fft_freqs_py.resize({(size_t)N});
        auto fft_buf = fft_freqs_py.request();
        double* ptr = (double*) fft_buf.ptr;
        for (int i = 0; i < N; i++){
            ptr[i] = fft_freqs[i];
        }
    }

    // optional COI
    py::array_t<double> coi_py(0);
    if (consider_coi) {
        std::vector<double> coi = compute_coi(N, dt);
        coi_py.resize({(size_t)N});
        auto coi_buf = coi_py.request();
        double* cptr = (double*) coi_buf.ptr;
        for (int i = 0; i < N; i++){
            cptr[i] = coi[i];
        }
    }

    // build final return
    if (consider_coi) {
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py, coi_py);
    } else {
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py);
    }
}

//------------------------------------------------------------------------------
// 3) local_gaussian_mean (optimized with binary search for 3σ window).
//------------------------------------------------------------------------------
py::array_t<double> local_gaussian_mean(
    py::array_t<double> signal,  // shape (N,) or (N,D)
    py::array_t<double> times,   // shape (N,) sorted ascending
    py::array_t<double> scales,  // shape (S,)
    double lam,
    bool use_omp
) {
    auto sig_buf = signal.request();
    auto time_buf= times.request();
    auto scl_buf = scales.request();

    int N = sig_buf.shape[0];
    int S = scl_buf.shape[0];
    int D = (sig_buf.ndim == 2) ? sig_buf.shape[1] : 1;
    if (sig_buf.ndim != 1 && sig_buf.ndim != 2) {
        throw std::runtime_error("signal must be 1D or 2D");
    }
    if (time_buf.ndim != 1 || time_buf.shape[0] != N) {
        throw std::runtime_error("times must be 1D, length N");
    }
    if (scl_buf.ndim != 1) {
        throw std::runtime_error("scales must be 1D");
    }

    const double* sig_ptr   = static_cast<const double*>(sig_buf.ptr);
    const double* time_ptr  = static_cast<const double*>(time_buf.ptr);
    const double* scale_ptr = static_cast<const double*>(scl_buf.ptr);

    // copy times into vector for binary search
    std::vector<double> times_vec(time_ptr, time_ptr + N);

    // output => shape (S, N, D)
    std::vector<py::ssize_t> out_shape = { S, N, D };
    py::array_t<double> B_out(out_shape);
    auto B_buf = B_out.request();
    double* B_ptr = static_cast<double*>(B_buf.ptr);

#ifdef _OPENMP
    if (use_omp) {
        // parallelize outer loop over scales
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < S; i++){
            double s = scale_ptr[i];
            double sigma = lam * s;
            double window = 3.0 * sigma;
            for (int n = 0; n < N; n++){
                double tn = times_vec[n];
                auto lb = std::lower_bound(times_vec.begin(), times_vec.end(), tn - window);
                auto ub = std::upper_bound(times_vec.begin(), times_vec.end(), tn + window);
                int lb_idx = std::distance(times_vec.begin(), lb);
                int ub_idx = std::distance(times_vec.begin(), ub);

                double sumW = 0.0;
                std::vector<double> accum(D, 0.0);

                for (int m = lb_idx; m < ub_idx; m++){
                    double dt_val = tn - times_vec[m];
                    double w = std::exp(- (dt_val*dt_val)/(2.0 * sigma * sigma));
                    sumW += w;
                    if (D == 1) {
                        accum[0] += sig_ptr[m] * w;
                    } else {
                        for (int d = 0; d < D; d++){
                            accum[d] += sig_ptr[m*D + d] * w;
                        }
                    }
                }
                for (int d = 0; d < D; d++){
                    accum[d] /= (sumW + 1e-30);
                    B_ptr[((i*N) + n)*D + d] = accum[d];
                }
            }
        }
    } else
#endif
    {
        // single-thread version
        for (int i = 0; i < S; i++){
            double s = scale_ptr[i];
            double sigma = lam * s;
            double window = 3.0 * sigma;
            for (int n = 0; n < N; n++){
                double tn = times_vec[n];
                auto lb = std::lower_bound(times_vec.begin(), times_vec.end(), tn - window);
                auto ub = std::upper_bound(times_vec.begin(), times_vec.end(), tn + window);
                int lb_idx = std::distance(times_vec.begin(), lb);
                int ub_idx = std::distance(times_vec.begin(), ub);

                double sumW = 0.0;
                std::vector<double> accum(D, 0.0);

                for (int m = lb_idx; m < ub_idx; m++){
                    double dt_val = tn - times_vec[m];
                    double w = std::exp(- (dt_val*dt_val)/(2.0 * sigma * sigma));
                    sumW += w;
                    if (D == 1) {
                        accum[0] += sig_ptr[m] * w;
                    } else {
                        for (int d = 0; d < D; d++){
                            accum[d] += sig_ptr[m*D + d] * w;
                        }
                    }
                }
                for (int d = 0; d < D; d++){
                    accum[d] /= (sumW + 1e-30);
                    B_ptr[((i*N) + n)*D + d] = accum[d];
                }
            }
        }
    }

    return B_out;
}

//------------------------------------------------------------------------------
// PYBIND11_MODULE
//------------------------------------------------------------------------------
PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "CWTPy: final single-file library with Morlet CWT and local Gaussian mean, avoiding fftw_init_threads for macOS stability.";

    // Morlet CWT
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
Returns a tuple:
  (W, scales, wavelet_freqs, psd_factor, fft_freqs[, coi]).
We do single-threaded FFT inside an OpenMP loop to avoid crashes on macOS.
)doc");

    // Local Gaussian Mean
    m.def("local_gaussian_mean", &local_gaussian_mean,
          py::arg("signal"),
          py::arg("times"),
          py::arg("scales"),
          py::arg("lam") = 1.0,
          py::arg("use_omp") = false,
R"doc(
Compute the local Gaussian mean of a signal (eqn(22)), restricting to ±3σ in time.
We do single-threaded summation in each thread to avoid conflicts with macOS + OpenMP.
)doc");
}
