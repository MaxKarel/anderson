#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//#include "helper_cuda.h"

#define _CUDA(x) checkCudaErrors(x)

__global__ void ker(float ** d_X, float * d_A, float * d_B, int n){
  printf("Addresses...\n");
  printf("dX    = %p\n", d_X);
  printf("dA    = %p\n", d_A);
  printf("dB    = %p\n", d_B);
  printf("dX[0] = %p\n", d_X[0]);
  printf("dX[0] = %p\n", d_X[1]);

  float * devA  = d_X[0];
  float * devB  = d_X[1];

  printf("\nValues...\n");
  for (int i=0; i<n; i++)
  printf("A[%d] = %f\n", i, devA[i]);
  for (int i=0; i<n; i++)
    printf("B[%d] = %f\n", i, devB[i]);

}

int main(void)
{
  /* Declarations */
  const int n = 10;
  const int nn = n * sizeof(float);
  float * h_A;
  float * h_B;
  float * d_A;
  float * d_B;
  float ** hst_ptr;

  /*
   * Allocate space for `h_A` and `h_B`
   */
  h_A = (float* )malloc(nn);
  h_B = (float* )malloc(nn);

  /*
   * Allocate space on the host for `hst_ptr`
   * as a mapped variable (so that the device can
   * access it directly)
   */
  _CUDA(  cudaHostAlloc((void**)&hst_ptr, 2*sizeof(float*), cudaHostAllocMapped) );

  for (int i=0; i<n; ++i){
    h_A[i] = i + 1.0f;
    h_B[i] = 20.0f + i;
  }

  /*
   * Allocate space on the device for `d_A` and `d_A`
   */
  _CUDA(  cudaMalloc((void**)&d_A, nn)  );
  _CUDA(  cudaMalloc((void**)&d_B, nn)  );

  _CUDA(  cudaMemcpy(d_A, h_A, nn, cudaMemcpyHostToDevice) );
  _CUDA(  cudaMemcpy(d_B, h_B, nn, cudaMemcpyHostToDevice) );

  hst_ptr[0]=d_A;
  hst_ptr[1]=d_B;

  ker<<<1,1>>>(hst_ptr, d_A, d_B, n);

  /* Free the resources.*/
  if (hst_ptr) _CUDA(  cudaFreeHost(hst_ptr)  );
  if (d_A) _CUDA(  cudaFree(d_A)  );
  if (d_A) _CUDA(  cudaFree(d_B)  );
  if (h_A) free(h_A);
  if (h_B) free(h_B);


  return 0;
}
