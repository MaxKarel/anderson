#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <thrust/complex.h>

using namespace std;

typedef thrust::complex<double> th_complex;

__global__ void cutrid_RC_1b(double *a,double *b,double *c,double *d,double *x) {
  int idx_global=blockIdx.x*blockDim.x+threadIdx.x;
  int idx=threadIdx.x;

  __shared__ double asub[4];
  __shared__ double bsub[4];
  __shared__ double csub[4];
  __shared__ double dsub[4];

  asub[idx]=a[idx_global];
  bsub[idx]=b[idx_global];
  csub[idx]=c[idx_global];
  dsub[idx]=d[idx_global];
  __syncthreads();
  //Reduction
  for(int stride=1;stride<4;stride*=2) {
    int margin_left=(idx-stride);
    int margin_right=(idx+stride);
    if(margin_left<0) margin_left=0;
    if(margin_right>=4) margin_right=3;
    double tmp1 = asub[idx] / bsub[margin_left];
    double tmp2 = csub[idx] / bsub[margin_right];
    double tmp3 = dsub[margin_right];
    double tmp4 = dsub[margin_left];
    __syncthreads();

    dsub[idx] = dsub[idx] - tmp4*tmp1-tmp3*tmp2;
    bsub[idx] = bsub[idx]-csub[margin_left]*tmp1-asub[margin_right]*tmp2;

    tmp3 = -csub[margin_right];
    tmp4 = -asub[margin_left];

    __syncthreads();
    asub[idx] = tmp3*tmp1;
    csub[idx] = tmp4*tmp2;
    __syncthreads();
  }

  x[idx_global]=dsub[idx]/bsub[idx];

}


int main () {
  int  n = 4;
	double a[4] = { 0, -1, -1, -1 };
	double b[4] = { 4,  4,  4,  4 };
	double c[4] = { -1, -1, -1, 0 };
	double d[4] = { 5,  5, 10, 23 };
	// results    { 2,  3,  5, 7  }
  double *d_a, *d_b, *d_c, *d_d, *d_x;
  cudaMalloc(&d_a,sizeof(double)*4);
  cudaMalloc(&d_b,sizeof(double)*4);
  cudaMalloc(&d_c,sizeof(double)*4);
  cudaMalloc(&d_d,sizeof(double)*4);
  cudaMalloc(&d_x,sizeof(double)*4);
  cudaMemcpy(d_a,a,sizeof(double)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,sizeof(double)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,c,sizeof(double)*4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d,d,sizeof(double)*4, cudaMemcpyHostToDevice);
	cutrid_RC_1b<<<4,4>>>(d_a,d_b,d_c,d_d,d_x);
  cudaMemcpy(d,d_x,sizeof(double)*4, cudaMemcpyDeviceToHost);
  cout.precision(17);
	for (int i = 0; i < n; i++) {
		cout << fixed << d[i] << endl;
	}
  return 0;
}
