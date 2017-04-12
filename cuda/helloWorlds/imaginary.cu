#include <iostream>
//#include <cuComplex.h>
#include <thrust/complex.h>

using namespace std;

__global__ void cuMultiply(thrust::complex<float>* A, thrust::complex<float>* B, int* size) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id<*size) {
    A[id] *= B[id];
    //A[id] *= 2.0; // (0, 1.0) * 2.0 == (0, 2.0)
  }
}


int main () {
  int N = 32;
  float sizefloat = sizeof(thrust::complex<float>);
  cout << "size of t::complex float is ==" << sizefloat << endl;
  cout << "while size of float is " << sizeof(float) << endl;
  int sizeArr = sizefloat*N;
  thrust::complex<float> a[N], b[N];
  thrust::complex<float> *d_a, *d_b;
  int* d_N;
  for (int i = 0; i<N; i++) {
    a[i] = thrust::complex<float>(0.0f, 1.0f);
    b[i] = thrust::complex<float>(0.0f, 1.0f);
  }

  cudaMalloc(&d_a, sizeArr);
  cudaMalloc(&d_b, sizeArr);
  cudaMalloc(&d_N, sizeof(int));
  cudaMemcpy(d_a, &a, sizeArr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeArr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);

  dim3 gridDim (1);
  dim3 blockDim (N);
  cuMultiply<<<gridDim ,blockDim>>>(d_a, d_b, d_N);

  cudaMemcpy(&a, d_a, sizeArr, cudaMemcpyDeviceToHost);
  for (int i= 0; i<N; i++) {
    cout << a[i];
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_N);

  return 0;
}
