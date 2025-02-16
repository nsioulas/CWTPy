// CWTPy/cwt_module.cpp
// CWTPy: A fast, cross‑platform CWT implementation using FFTW and OpenMP.
// This module computes the continuous wavelet transform using an L2‑normalized
// Morlet wavelet and returns the coefficients, scales, and frequencies.
// 
// Author: Nikos Sioulas (Space Sciences Laboratory, UC Berkeley)

#include <Python.h>    // Include full Python API first!
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

static inline double morlet_factor() {
    return std::pow(M_PI, -0.25);
}
static inline double freq_to_scale(double freq, double omega0) {
    double corr = 1.0 + 1.0/(2.0*omega0*omega0);
    if (freq <= 0.0) throw std::runtime_error("freq_to_scale: freq <= 0");
    return (omega0/(2.0*M_PI*freq))*corr;
}
static inline double scale_to_freq(double s, double omega0) {
    double corr = 1.0 + 1.0/(2.0*omega0*omega0);
    return (omega0/(2.0*M_PI*s))*corr;
}
static std::vector<double> make_scales_log(double s0, double s1, int nv) {
    if (s0<=0||s1<=s0) throw std::runtime_error("Invalid scale range");
    double a = std::pow(2.0, 1.0/double(nv));
    std::vector<double> scales;
    double s=s0;
    while (s<=s1) {
        scales.push_back(s);
        s*=a;
    }
    return scales;
}

py::tuple cwt_morlet_full(
    py::array_t<double> signal,
    double dt,
    int nv,
    double omega0,
    double min_freq,
    double max_freq,
    bool use_omp
) {
    auto buf = signal.request();
    if (buf.ndim !=1)
        throw std::runtime_error("signal must be 1D");
    int N=buf.shape[0];
    if (N<2)
        throw std::runtime_error("Signal too short");

    double* sig_ptr = (double*) buf.ptr;
    double fs=1.0/dt;

    if (max_freq<=0.0) max_freq=fs/2.0;
    if (min_freq<=0.0) min_freq=1.0/(N*dt);
    if (min_freq>=max_freq)
        throw std::runtime_error("min_freq >= max_freq => invalid range");

    double s0 = freq_to_scale(max_freq, omega0);
    double s1 = freq_to_scale(min_freq, omega0);

    std::vector<double> scales = make_scales_log(s0, s1, nv);
    int num_scales=(int)scales.size();

    fftw_complex* in=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex* out=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    for (int i=0;i<N;i++){
        in[i][0]=sig_ptr[i];
        in[i][1]=0.0;
    }
    fftw_plan plan_forward=fftw_plan_dft_1d(N,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    std::vector<std::complex<double>> sig_fft(N);
    for(int i=0;i<N;i++){
        sig_fft[i]={out[i][0],out[i][1]};
    }
    fftw_destroy_plan(plan_forward);
    fftw_free(in);

    fftw_complex* freq_prod=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex* inv=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_plan plan_backward=fftw_plan_dft_1d(N,freq_prod,inv,FFTW_BACKWARD,FFTW_ESTIMATE);

    std::vector<std::complex<double>> W_data(num_scales*N);
    double norm_factor=morlet_factor();

    std::vector<double> omega_vec(N);
    double df=fs/double(N);
    for(int k=0;k<N;k++){
        double freq_k=(k<=N/2)? k*df : -(N-k)*df;
        omega_vec[k]=2.0*M_PI*freq_k;
    }

#ifdef _OPENMP
    if(use_omp){
        #pragma omp parallel for
        for(int sidx=0;sidx<num_scales;sidx++){
            double s=scales[sidx];
            for(int k=0;k<N;k++){
                double arg=s*omega_vec[k]-omega0;
                double wavelet=std::exp(-0.5*arg*arg)*std::sqrt(s)*norm_factor;
                auto val=sig_fft[k]*wavelet;
                freq_prod[k][0]=val.real();
                freq_prod[k][1]=val.imag();
            }
            fftw_execute(plan_backward);
            for(int n=0;n<N;n++){
                double re=inv[n][0]/double(N);
                double im=inv[n][1]/double(N);
                W_data[sidx*N+n]={re,im};
            }
        }
    } else
#endif
    {
        for(int sidx=0;sidx<num_scales;sidx++){
            double s=scales[sidx];
            for(int k=0;k<N;k++){
                double arg=s*omega_vec[k]-omega0;
                double wavelet=std::exp(-0.5*arg*arg)*std::sqrt(s)*norm_factor;
                auto val=sig_fft[k]*wavelet;
                freq_prod[k][0]=val.real();
                freq_prod[k][1]=val.imag();
            }
            fftw_execute(plan_backward);
            for(int n=0;n<N;n++){
                double re=inv[n][0]/double(N);
                double im=inv[n][1]/double(N);
                W_data[sidx*N+n]={re,im};
            }
        }
    }

    fftw_destroy_plan(plan_backward);
    fftw_free(freq_prod);
    fftw_free(inv);
    fftw_free(out);

    std::vector<double> freq_arr(num_scales);
    for(int i=0;i<num_scales;i++){
        freq_arr[i]=scale_to_freq(scales[i], omega0);
    }

    py::array_t<std::complex<double>> W_py({num_scales,N});
    auto W_buf=W_py.request();
    auto* W_ptr=(std::complex<double>*) W_buf.ptr;
    std::memcpy(W_ptr,W_data.data(),num_scales*N*sizeof(std::complex<double>));

    py::array_t<double> scales_py(num_scales);
    auto s_buf=scales_py.request();
    double* s_ptr=(double*) s_buf.ptr;
    for(int i=0;i<num_scales;i++){
        s_ptr[i]=scales[i];
    }

    py::array_t<double> freqs_py(num_scales);
    auto f_buf=freqs_py.request();
    double* f_ptr=(double*) f_buf.ptr;
    for(int i=0;i<num_scales;i++){
        f_ptr[i]=freq_arr[i];
    }

    return py::make_tuple(W_py, scales_py, freqs_py);
}

PYBIND11_MODULE(cwt_module, m){
    m.doc()="CWTPy: A fast Morlet CWT using FFTW and optional OpenMP.";
    m.def("cwt_morlet_full", &cwt_morlet_full,
          py::arg("signal"),
          py::arg("dt"),
          py::arg("nv")       = 32,
          py::arg("omega0")   = 6.0,
          py::arg("min_freq") = 0.0,
          py::arg("max_freq") = 0.0,
          py::arg("use_omp")  = false,
          R"doc(
Compute CWT with an L2-normalized Morlet wavelet.

Parameters
----------
signal   : 1D np.ndarray
dt       : float
nv       : int, voices per octave
omega0   : float, Morlet parameter
min_freq : float
max_freq : float
use_omp  : bool

Returns
-------
W, scales, freqs
)doc");
}
