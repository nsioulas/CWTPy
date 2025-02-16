// cwt_module.cpp
// pyCWTfast: A fast, cross‑platform CWT implementation using FFTW and OpenMP.
// This module computes the continuous wavelet transform using an L2‐normalized
// Morlet wavelet and returns the coefficients, scales, and frequencies.
// 
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    //

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fftw3.h>


#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

// If you want OpenMP:
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

// L2 normalization factor for Morlet (pi^(-1/4)).
static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}

// eqn(12): freq = (omega0/(2π s)) * (1 + 1/(2ω0^2)).
static inline double scale_to_freq(double s, double omega0) {
    
    //  double corr = (1.0 + 1.0/(2.0 * omega0 * omega0));
    double corr = 1.0;
    return (omega0/(2.0*M_PI*s)) * corr;
}

// eqn(12) inverted: s = (omega0/(2π freq)) * (1 + 1/(2ω0^2)).
static inline double freq_to_scale(double freq, double omega0) {
    if (freq <= 0.0)
        throw std::runtime_error("freq_to_scale: frequency must be > 0");
   //  double corr = (1.0 + 1.0/(2.0 * omega0 * omega0));
    double corr = 1.0;
    return (omega0/(2.0*M_PI*freq)) * corr;
}

// Build log-spaced scales from s0 -> s1, 2^(1/nv) per octave.
std::vector<double> make_scales_log(double s0, double s1, int nv) {
    if (s0 <= 0 || s1 <= s0)
        throw std::runtime_error("make_scales_log: invalid scale range");
    double a = std::pow(2.0, 1.0/double(nv));
    std::vector<double> scales;
    for (double s = s0; s <= s1; s *= a) {
        scales.push_back(s);
    }
    return scales;
}

/*
   cwt_morlet_full:
   ----------------
   signal:   real 1D array
   dt:       sampling interval
   nv:       voices per octave
   omega0:   Morlet parameter
   min_freq, max_freq:
       If <= 0 => default to [1/(N dt), fs/2].
   use_omp:  if true => parallelize over scales with separate plans/buffers

   returns (W, scales, freqs):

   W:      shape = (num_scales, N) [complex64 or complex128 in Python]
   scales: length = num_scales
   freqs:  length = num_scales
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
    //----- 1) Parse input, define freq range
    auto buf = signal.request();
    if (buf.ndim != 1)
        throw std::runtime_error("signal must be 1D");
    int N = buf.shape[0];
    if (N < 2)
        throw std::runtime_error("Signal too short.");

    const double* sig_ptr = (double*) buf.ptr;
    double fs = 1.0 / dt;

    // if user didn't supply freq bounds, pick defaults
    if (max_freq <= 0.0)
        max_freq = fs/2.0;              // near Nyquist
    if (min_freq <= 0.0)
        min_freq = 1.0 / (N * dt);      // ~lowest resolvable freq
    if (min_freq >= max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range.");

    // Convert freq range -> scale range
    double smin = freq_to_scale(max_freq, omega0); // smallest scale
    double smax = freq_to_scale(min_freq, omega0); // largest scale
    if (smin >= smax)
        throw std::runtime_error("Scale range is invalid. Check freq bounds.");

    // Build log-spaced scale array
    std::vector<double> scales = make_scales_log(smin, smax, nv);
    int num_scales = (int) scales.size();

    //----- 2) Forward FFT of the signal
    fftw_complex* in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);

    // fill input for FFT
    for (int i=0; i<N; i++) {
        in[i][0] = sig_ptr[i];
        in[i][1] = 0.0;
    }
    fftw_plan plan_fwd = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_fwd);
    fftw_destroy_plan(plan_fwd);
    fftw_free(in);

    // store FFT in a std::vector
    std::vector<std::complex<double>> sig_fft(N);
    for (int i=0; i<N; i++) {
        sig_fft[i] = std::complex<double>(out[i][0], out[i][1]);
    }
    fftw_free(out);

    //----- 3) Prepare output array for W: shape (num_scales, N)
    std::vector<std::complex<double>> W_data(num_scales*N);

    // Angular frequency array
    std::vector<double> omega_vec(N);
    double df = fs/double(N);
    for (int k=0; k<N; k++) {
        double f_k = 0.0;
        if (k <= N/2)
            f_k = k*df;
        else
            f_k = -(N - k)*df;
        omega_vec[k] = 2.0*M_PI * f_k;
    }

    double norm = morlet_factor(); // pi^(-1/4)

    //----- 4) Parallel block: each thread has its own freq buffers & plan
#ifdef _OPENMP
    if (use_omp) {
        #pragma omp parallel
        {
            // Each thread allocates freq_prod & inv & plan
            fftw_complex* freq_prod_local = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
            fftw_complex* inv_local       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
            fftw_plan plan_bwd_local = fftw_plan_dft_1d(N, freq_prod_local, inv_local,
                                                        FFTW_BACKWARD, FFTW_ESTIMATE);

            #pragma omp for schedule(static)
            for (int sidx=0; sidx<num_scales; sidx++) {
                double s = scales[sidx];
                // freq-domain multiplication
                for (int k=0; k<N; k++) {
                    double w = omega_vec[k];
                    double arg = s*w - omega0;
                    double wavelet = std::exp(-0.5*arg*arg)*std::sqrt(s)*norm;
                    std::complex<double> val = sig_fft[k]*wavelet;
                    freq_prod_local[k][0] = val.real();
                    freq_prod_local[k][1] = val.imag();
                }
                // iFFT
                fftw_execute(plan_bwd_local);
                // store => divide by N
                for (int n=0; n<N; n++) {
                    double re = inv_local[n][0]/double(N);
                    double im = inv_local[n][1]/double(N);
                    W_data[sidx*N + n] = std::complex<double>(re, im);
                }
            }

            fftw_destroy_plan(plan_bwd_local);
            fftw_free(freq_prod_local);
            fftw_free(inv_local);
        } // end parallel region
    } 
    else
#endif
    {
        // Non-OMP version (single-thread)
        // allocate freq_prod & inv once
        fftw_complex* freq_prod = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        fftw_complex* inv       = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
        fftw_plan plan_bwd = fftw_plan_dft_1d(N, freq_prod, inv, FFTW_BACKWARD, FFTW_ESTIMATE);

        for (int sidx=0; sidx<num_scales; sidx++) {
            double s = scales[sidx];
            for (int k=0; k<N; k++) {
                double w = omega_vec[k];
                double arg = s*w - omega0;
                double wavelet = std::exp(-0.5*arg*arg)*std::sqrt(s)*norm;
                std::complex<double> val = sig_fft[k]*wavelet;
                freq_prod[k][0] = val.real();
                freq_prod[k][1] = val.imag();
            }
            fftw_execute(plan_bwd);
            for (int n=0; n<N; n++) {
                double re = inv[n][0]/double(N);
                double im = inv[n][1]/double(N);
                W_data[sidx*N + n] = std::complex<double>(re, im);
            }
        }

        fftw_destroy_plan(plan_bwd);
        fftw_free(freq_prod);
        fftw_free(inv);
    }

    //----- 5) Build the freqs array
    std::vector<double> freqs(num_scales);
    for (int i=0; i<num_scales; i++) {
        freqs[i] = scale_to_freq(scales[i], omega0);
    }

    //----- 6) Convert to Python objects
    py::array_t<std::complex<double>> W_py({num_scales, N});
    {
        auto W_buf = W_py.request();
        auto* W_ptr = (std::complex<double>*)W_buf.ptr;
        std::memcpy(W_ptr, W_data.data(), num_scales*N*sizeof(std::complex<double>));
    }


    // Option A: Use a single integer
    py::array_t<double> scales_py(num_scales);
    py::array_t<double> freqs_py(num_scales);

    {
        auto s_buf = scales_py.request();
        auto f_buf = freqs_py.request();
        double* s_ptr = (double*) s_buf.ptr;
        double* f_ptr = (double*) f_buf.ptr;
        for (int i=0; i<num_scales; i++) {
            s_ptr[i] = scales[i];
            f_ptr[i] = freqs[i];
        }
    }

    // return (W, scales, freqs)
    return py::make_tuple(W_py, scales_py, freqs_py);
}

PYBIND11_MODULE(cwt_module, m) {
    m.doc() = "Faster Morlet CWT with separate iFFT plans per thread to improve OpenMP scaling";

    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv")=32,
          py::arg("omega0")=6.0,
          py::arg("min_freq")=0.0,
          py::arg("max_freq")=0.0,
          py::arg("use_omp")=false,
R"doc(
Compute Morlet CWT of a real 1D signal with optional parallelization.
Returns (W, scales, freqs).
)doc");
}
