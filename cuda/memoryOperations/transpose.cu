#include <iostream>
#include <fstream>
#include <sys/resource.h>
#include <cmath>
#include <ctime>
#include <complex>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <cublas_v2.h>

using namespace std;

typedef thrust::complex<double> th_complex;

int nIter = 1024; //Number of iteration fo each transpose method
const int xmax = 2048; //size of arrays
const int TILE_DIM = 32;
th_complex h_u[xmax][xmax];
th_complex h_uH[xmax][xmax];
th_complex h_ub[xmax][xmax];
th_complex h_ua[xmax][xmax];

bool checkResult(th_complex h_u[][xmax], th_complex h_uH[][xmax]);
void initializeHostArrays(th_complex h_u[][xmax]);

__global__ void transposeCU(th_complex* d_u,
                            th_complex* d_uH,int xmax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < xmax)&(j<xmax)) {
    d_uH[i*xmax+j] = d_u[j*xmax+i];
  }
}
/*
__global__ void transposeCoalesced(th_complex *odata, th_complex *idata)
{
  /////////////////////////////////////////////////////////////////
  ///
  /// Code is from nVidia website : https://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
  ///
  /////////////////////////////////////////////////////////////////

  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += )
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}
*/
int main() {
  cout << "Starting the test ...." << endl;

  // const rlim_t kStackSize = 16 * 1024 * 1024;   // min stack size = 16 MB
  //   struct rlimit rl;
  //   int result;
  //
  //   result = getrlimit(RLIMIT_STACK, &rl);
  //   if (result == 0)
  //   {
  //       if (rl.rlim_cur < kStackSize)
  //       {
  //           rl.rlim_cur = kStackSize;
  //           result = setrlimit(RLIMIT_STACK, &rl);
  //           if (result != 0)
  //           {
  //               fprintf(stderr, "setrlimit returned result = %d\n", result);
  //           }
  //       }
  //   }

  initializeHostArrays(h_u);
  if((nIter%2) != 0) {nIter+=1;} //it has to be even in order to pass checkResult()

  cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
  const cuDoubleComplex* _alpha = &alpha;
  const cuDoubleComplex* _beta = &beta;
  cublasHandle_t handle;
  cublasCreate(&handle);

  double arrSize = sizeof(th_complex) * xmax * xmax;

  th_complex *d_u, *d_uH;
  cout << "Allocating arrays on device" << endl;
  cudaMalloc(&d_u, arrSize);
  cudaMalloc(&d_uH, arrSize);

  cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uH, h_uH, arrSize, cudaMemcpyHostToDevice);

  cuDoubleComplex* d_uHc = reinterpret_cast<cuDoubleComplex* >(d_uH);
  cuDoubleComplex* d_uc = reinterpret_cast<cuDoubleComplex* >(d_u);

  cout << "Address for thrust arrays d_u and d_uH :" <<endl;
  cout << "th_compl: " << d_u << " cuCompl: " << d_uc << endl;
  cout << "th_compl: " << d_uH << " cuCompl: " << d_uHc << endl;

  dim3 dimBlock(32,32);
  dim3 dimGrid(xmax / 32 + 1, xmax / 32 + 1);
  /////////////////////////////////
  ////    cuBLAS_v2            ////
  /////////////////////////////////

  cout << "Starting cuBLAS_v2" <<endl;
  clock_t start = clock();
  for (int _t = 0; _t< nIter; _t++) {
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                    _alpha, d_uc, xmax,
                    _beta, d_uc, xmax,
                    d_uHc, xmax);
    if (cudaMemcpy(d_u, d_uH, arrSize, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      cout << "Error Copying at cuBLAS_v2" << endl;
    }
  }
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  clock_t end = clock();
  double t = double(end - start) / CLOCKS_PER_SEC;
  if (checkResult(h_u, h_uH)) {
    cout << "cublasZgeam PASSED:" <<endl;
    cout << "time = " << t <<endl << endl;
  }
  else {cout << "cublasZgeam FAILED" << endl;}

  /////////////////////////////////
  ////    Simple Transpose     ////
  /////////////////////////////////
  cout<< "Starting Simple Transpose" << endl;
  start = clock();
  for (int _t = 0; _t< nIter; _t++) {
    transposeCU<<<dimGrid,dimBlock>>>(d_u, d_uH, xmax);
    cudaError_t lastErr = cudaGetLastError();
    if ( cudaSuccess != lastErr ) {
      cout << lastErr << endl;
    }
    if (cudaMemcpy(d_uH, d_u, arrSize, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      cout << "Error Copying" << endl;
    }
  }
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  end = clock();
  t = double(end - start) / CLOCKS_PER_SEC;
  if (checkResult(h_u, h_uH)) {
    cout << "Simple transpose PASSED:" <<endl;
    cout << "time = " << t <<endl;
  }
  else {cout << "Simple transpose FAILED" << endl;}

  /////////////////////////////////
  ////  Coalesced Transpose    ////
  /////////////////////////////////
  cout<< "Starting Coalesced Transpose (shared memory)" << endl;
  start = clock();
  for (int _t = 0; _t< nIter; _t++) {
    //transposeCoalesced<<<dimGrid,dimBlock>>>(d_u, d_uH);
    cudaError_t lastErr = cudaGetLastError();
    if ( cudaSuccess != lastErr ) {
      cout << lastErr << endl;
    }
    if (cudaMemcpy(d_uH, d_u, arrSize, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      cout << "Error Copying" << endl;
    }
  }
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  end = clock();
  t = double(end - start) / CLOCKS_PER_SEC;
  if (checkResult(h_u, h_uH)) {
    cout << "Coalesced Transpose (shared memory) PASSED:" <<endl;
    cout << "time = " << t <<endl;
  }
  else {cout << "Coalesced Transpose (shared memory) FAILED" << endl;}
  cudaFree(d_u);
  cudaFree(d_uH);

  return 0;
}
void initializeHostArrays(th_complex h_u[][xmax]) {
  th_complex imag_one (0.0,1.0);
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j]=th_complex(7.0f*(float)(random()%2000/2000.0-1.0f), 0.0f);
      h_u[i][j] *= -imag_one;
    }
  }
}

bool checkResult(th_complex h_u[][xmax], th_complex h_uH[][xmax]) {
  for(int i = 0; i<xmax; i++) {
    for(int j = 0; j<xmax; j++) {
      if(h_u[i][j]!=h_uH[i][j]) {
        cout << "ERROR: values dont match!" << endl;
        cout << "(" << i << "," << j << "):" << h_u[i][j] << "!=" << h_uH[i][j] << endl;
        return false;
      }
    }
  }
  return true;
}
