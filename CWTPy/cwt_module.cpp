// CWTPy/cwt_module.cpp
// CWTPy: A single unified C++ module for:
//   1) cwt_morlet_full  : Morlet wavelet continuous wavelet transform (CWT)
//   2) local_gaussian_mean : Local Gaussian mean (Equation (22)) with normalization
//
// This version supports user-specified scale distributions via a new parameter "scale_type".
// Supported options include:
//   - "log": Exponentially spaced scales.
//   - "log-piecewise": Log spacing with downsampling at high scales.
//   - "linear": Linearly spaced scales.
// It uses FFTW_MEASURE for optimized planning and pre-creates FFTW backward plans (serially)
// before executing them concurrently with OpenMP. The inner loop over frequency bins is annotated
// with OpenMP SIMD hints to encourage vectorization.
//
// IMPORTANT: To use FFTW's OpenMP routines, compile and link against the FFTW OpenMP library (e.g. fftw3_omp).
// On macOS, ensure FFTW is installed with OpenMP support (via Homebrew) and update your setup.py accordingly.
//
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    // Must be included first!
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
#include <string>

namespace py = pybind11;

//------------------------------------------------------------------------------
// Helper functions: compute_fft_freqs and compute_coi
//------------------------------------------------------------------------------
static std::vector<double> compute_fft_freqs(int N, double fs) {
    std::vector<double> fft_freqs(N);
    double df = fs / double(N);
    for (int k = 0; k < N; k++){
        if (k <= N/2)
            fft_freqs[k] = k * df;
        else
            fft_freqs[k] = -(N - k) * df;
    }
    return fft_freqs;
}

static std::vector<double> compute_coi(int N, double dt) {
    std::vector<double> coi(N);
    for (int t = 0; t < N; t++) {
        double d = std::min(double(t+1), double(N-t));
        coi[t] = dt * std::sqrt(2.0) * d;
    }
    return coi;
}

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
        if (i == 0 || i == nsteps)
            sum += f;
        else if (i % 2 == 1)
            sum += 4.0 * f;
        else
            sum += 2.0 * f;
    }
    double integral = (h / 3.0) * sum;
    return integral / std::sqrt(M_PI);
}

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

static std::vector<double> make_scales_log(double s0, double s1, int nv) {
    if (s0 <= 0 || s1 <= s0)
        throw std::runtime_error("make_scales_log: invalid scale range");
    double a = std::pow(2.0, 1.0/double(nv));
    std::vector<double> scales;
    for (double s = s0; s <= s1; s *= a)
        scales.push_back(s);
    return scales;
}

// Generate scales based on a preset string.
static std::vector<double> generate_scales(double s0, double s1, int nv, const std::string &scale_type) {
    if (scale_type == "log") {
        return make_scales_log(s0, s1, nv);
    } else if (scale_type == "log-piecewise") {
        std::vector<double> scales = make_scales_log(s0, s1, nv);
        double threshold = 1.05;  // if consecutive scales have ratio < threshold, skip
        std::vector<double> filtered;
        filtered.push_back(scales[0]);
        for (size_t i = 1; i < scales.size(); i++) {
            if (scales[i] / filtered.back() > threshold)
                filtered.push_back(scales[i]);
        }
        return filtered;
    } else if (scale_type == "linear") {
        // Create a fixed number of linearly spaced scales.
        double octaves = std::log2(s1 / s0);
        int count = std::max(2, (int)std::ceil(octaves * nv));
        std::vector<double> scales(count);
        double step = (s1 - s0) / (count - 1);
        for (int i = 0; i < count; i++) {
            scales[i] = s0 + i * step;
        }
        return scales;
    } else {
        // Default to log-piecewise.
        return make_scales_log(s0, s1, nv);
    }
}

//------------------------------------------------------------------------------
// 2) cwt_morlet_full
//------------------------------------------------------------------------------
/*
   cwt_morlet_full:
   Computes the Morlet continuous wavelet transform (CWT) of a real 1D signal.
   Returns a tuple:
     (W, scales, wave_freqs, psd_factor, fft_freqs[, coi])
   where:
     - W is the array of CWT coefficients (num_scales x N),
     - scales is the vector of scales (in seconds),
     - wave_freqs are the corresponding wavelet frequencies (Hz),
     - psd_factor = 4π/(C ω₀), computed via Simpson's rule,
     - fft_freqs are the FFT frequencies (Hz),
     - coi is the cone-of-influence (if requested).
     
   New parameter:
     scale_type: string specifying scale distribution. Options: "log", "log-piecewise", "linear".
     Default is "log-piecewise".
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
    const std::string &scale_type,
    bool consider_coi,
    bool return_fft
) {
#ifdef _OPENMP
    if (use_omp) {
        if (!fftw_init_threads())
            throw std::runtime_error("fftw_init_threads() failed");
        int num_threads = omp_get_max_threads();
        fftw_plan_with_nthreads(num_threads);
    }
#endif

    auto buf = signal.request();
    if (buf.ndim != 1)
        throw std::runtime_error("signal must be 1D");
    int N = buf.shape[0];
    if (N < 2)
        throw std::runtime_error("Signal too short");
    const double* sig_ptr = static_cast<const double*>(buf.ptr);

    double fs = 1.0/dt;
    if (max_freq <= 0.0)
        max_freq = fs/2.0;
    if (min_freq <= 0.0)
        min_freq = 1.0/(N*dt);
    if (min_freq >= max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range");

    double smin = freq_to_scale(max_freq, omega0);  // smallest scale
    double smax = freq_to_scale(min_freq, omega0);    // largest scale
    if (smin >= smax)
        throw std::runtime_error("Scale range is invalid. Check freq bounds.");
    std::vector<double> scales = generate_scales(smin, smax, nv, scale_type);
    int num_scales = static_cast<int>(scales.size());

    // Forward FFT.
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    for (int i = 0; i < N; i++){
        in[i][0] = sig_ptr[i];
        in[i][1] = 0.0;
    }
    fftw_plan plan_fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_execute(plan_fwd);
    fftw_destroy_plan(plan_fwd);
    fftw_free(in);
    std::vector<std::complex<double>> sig_fft(N);
    for (int i = 0; i < N; i++){
        sig_fft[i] = std::complex<double>(out[i][0], out[i][1]);
    }
    fftw_free(out);

    // Allocate container for CWT coefficients.
    std::vector<std::complex<double>> W_data(num_scales * N);

    // Build angular frequency array.
    std::vector<double> omega_vec(N);
    double df_val = fs / double(N);
    for (int k = 0; k < N; k++){
        double f_k = (k <= N/2) ? k * df_val : -(N - k) * df_val;
        omega_vec[k] = 2.0 * M_PI * f_k;
    }
    double norm = morlet_factor() * norm_mult;

    // Pre-create FFTW backward plans and associated buffers (serially).
    std::vector<fftw_plan> bwd_plans(num_scales);
    std::vector<fftw_complex*> freq_prods(num_scales);
    std::vector<fftw_complex*> inv_buffers(num_scales);
    for (int sidx = 0; sidx < num_scales; sidx++){
        freq_prods[sidx] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        inv_buffers[sidx] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        bwd_plans[sidx] = fftw_plan_dft_1d(N, freq_prods[sidx], inv_buffers[sidx], FFTW_BACKWARD, FFTW_MEASURE);
    }

    // Parallel execution: fill buffers and execute plans concurrently.
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(use_omp)
#endif
    for (int sidx = 0; sidx < num_scales; sidx++){
        double s = scales[sidx];
        double sqrt_s = std::sqrt(s); // precompute √s
        for (int k = 0; k < N; k++){
            double arg = s * omega_vec[k] - omega0;
            double wavelet = std::exp(-0.5 * arg * arg) * sqrt_s * norm;
            std::complex<double> val = sig_fft[k] * wavelet;
            freq_prods[sidx][k][0] = val.real();
            freq_prods[sidx][k][1] = val.imag();
        }
        fftw_execute(bwd_plans[sidx]);
        for (int n = 0; n < N; n++){
            double re = inv_buffers[sidx][n][0] / double(N);
            double im = inv_buffers[sidx][n][1] / double(N);
            W_data[sidx * N + n] = std::complex<double>(re, im);
        }
    }

    // Clean up FFTW backward plans.
    for (int sidx = 0; sidx < num_scales; sidx++){
        fftw_destroy_plan(bwd_plans[sidx]);
        fftw_free(freq_prods[sidx]);
        fftw_free(inv_buffers[sidx]);
    }

    // Compute wavelet frequencies (in Hz).
    std::vector<double> wave_freqs(num_scales);
    for (int i = 0; i < num_scales; i++){
        wave_freqs[i] = scale_to_freq(scales[i], omega0);
    }

    // Partial PSD factor: 4π/(C ω₀)
    double C_val = compute_admissibility(omega0);
    double psd_factor = (4.0 * M_PI) / (C_val * omega0);

    // Convert outputs to Python arrays.
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
    py::array_t<double> fft_freqs_py(0);
    if (return_fft){
        std::vector<double> fft_freqs = compute_fft_freqs(N, fs);
        fft_freqs_py.resize({(size_t)N});
        auto fft_buf = fft_freqs_py.request();
        double* ptr = (double*) fft_buf.ptr;
        for (int i = 0; i < N; i++){
            ptr[i] = fft_freqs[i];
        }
    }
    py::array_t<double> coi_py(0);
    if (consider_coi){
        std::vector<double> coi = compute_coi(N, dt);
        coi_py.resize({(size_t)N});
        auto coi_buf = coi_py.request();
        double* cptr = (double*) coi_buf.ptr;
        for (int i = 0; i < N; i++){
            cptr[i] = coi[i];
        }
    }

    if (consider_coi)
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py, coi_py);
    else
        return py::make_tuple(W_py, scales_py, freqs_py, psd_factor, fft_freqs_py);
}

//------------------------------------------------------------------------------
// 3) local_gaussian_mean (optimized with binary search)
//------------------------------------------------------------------------------
/*
   local_gaussian_mean:
   Computes the local Gaussian mean B_n(s) for each scale s and time index n:
     B_n(s) = [ Σ_{m in window} B_m exp( - (t_n-t_m)^2/(2 lam^2 s^2) ) ]
              / [ Σ_{m in window} exp( - (t_n-t_m)^2/(2 lam^2 s^2) ) ]
   Only considers m for which |t_n - t_m| <= 3*lam*s.
   Assumes times is sorted in ascending order.
   Returns an array of shape (num_scales, N, D) for a D-component signal (or (S,N) for 1D).
*/
py::array_t<double> local_gaussian_mean(
    py::array_t<double> signal,
    py::array_t<double> times,
    py::array_t<double> scales,
    double lam,
    bool use_omp
) {
    auto sig_buf = signal.request();
    auto time_buf = times.request();
    auto scale_buf = scales.request();

    int N = sig_buf.shape[0];
    int S = scale_buf.shape[0];
    int D = (sig_buf.ndim == 2) ? sig_buf.shape[1] : 1;
    if (sig_buf.ndim != 1 && sig_buf.ndim != 2)
        throw std::runtime_error("signal must be 1D or 2D");
    if (time_buf.ndim != 1 || time_buf.shape[0] != N)
        throw std::runtime_error("times must be 1D, length N");
    if (scale_buf.ndim != 1)
        throw std::runtime_error("scales must be 1D");

    const double* sig_ptr = static_cast<const double*>(sig_buf.ptr);
    const double* time_ptr = static_cast<const double*>(time_buf.ptr);
    const double* scale_ptr = static_cast<const double*>(scale_buf.ptr);

    // Copy times into vector for binary search.
    std::vector<double> times_vec(time_ptr, time_ptr + N);

    // Output shape: (S, N, D)
    std::vector<py::ssize_t> out_shape = { S, N, D };
    py::array_t<double> B_out(out_shape);
    auto B_buf = B_out.request();
    double* B_ptr = static_cast<double*>(B_buf.ptr);

#ifdef _OPENMP
    if (use_omp) {
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
                    double w = std::exp(- (dt_val * dt_val) / (2.0 * sigma * sigma));
                    sumW += w;
                    if (D == 1)
                        accum[0] += sig_ptr[m] * w;
                    else {
                        for (int d = 0; d < D; d++){
                            accum[d] += sig_ptr[m * D + d] * w;
                        }
                    }
                }
                for (int d = 0; d < D; d++){
                    accum[d] /= (sumW + 1e-30);
                    B_ptr[((i * N) + n) * D + d] = accum[d];
                }
            }
        }
    } else
#endif
    {
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
                    double w = std::exp(- (dt_val * dt_val) / (2.0 * sigma * sigma));
                    sumW += w;
                    if (D == 1)
                        accum[0] += sig_ptr[m] * w;
                    else {
                        for (int d = 0; d < D; d++){
                            accum[d] += sig_ptr[m * D + d] * w;
                        }
                    }
                }
                for (int d = 0; d < D; d++){
                    accum[d] /= (sumW + 1e-30);
                    B_ptr[((i * N) + n) * D + d] = accum[d];
                }
            }
        }
    }

    return B_out;
}

PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "CWTPy: final single-file library with Morlet CWT and local Gaussian mean, supporting user-specified scale distribution.";
    
    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv") = 32,
          py::arg("omega0") = 6.0,
          py::arg("min_freq") = 0.0,
          py::arg("max_freq") = 0.0,
          py::arg("use_omp") = false,
          py::arg("norm_mult") = 1.0,
          py::arg("scale_type") = std::string("log-piecewise"),
          py::arg("consider_coi") = false,
          py::arg("return_fft") = false,
R"doc(
Compute the Morlet continuous wavelet transform (CWT) of a real 1D signal.
Parameters:
  - scale_type: one of {"log", "log-piecewise", "linear"}. 
    "log": exponentially spaced scales,
    "log-piecewise": exponential spacing with downsampling at high scales,
    "linear": linearly spaced scales.
Returns a tuple:
  (W, scales, wavelet_freqs, psd_factor, fft_freqs[, coi]),
where psd_factor = 4π/(C ω₀), computed numerically.
)doc");

    m.def("local_gaussian_mean", &local_gaussian_mean,
          py::arg("signal"),
          py::arg("times"),
          py::arg("scales"),
          py::arg("lam") = 1.0,
          py::arg("use_omp") = false,
R"doc(
Compute the local Gaussian mean of a signal (eqn(22)) with 3σ-window optimization.
For each scale s and time index n, compute:
  B_n(s) = [ Σ_{m in window} B_m exp( - (t_n - t_m)^2/(2*lam^2*s^2) ) ]
           / [ Σ_{m in window} exp( - (t_n - t_m)^2/(2*lam^2*s^2) ) ]
Returns an array of shape (num_scales, N, D) (or (num_scales, N) for 1D signals).
)doc");
}
