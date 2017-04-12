#include <iostream>
//#include <cublas_v2.h>
using namespace std;

__global__ void CUDAadd(int *a, int *b, int count) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id<count) {
    a[id] += b[id];
  }
}


int main() {
  const int count = 1027;
  int sizeInt = sizeof(int);
  int sizeArr = sizeInt*count;
  int a[count], b[count], c[count]; // host copies of a, b
  int *d_a, *d_b, *d_c; // device copies of a, b
  cout << "d_a(prior to Malloc)=" << d_a <<endl;
  // Allocate space for device copies of a, b, c
  //if (cudaMalloc((void **)&d_a, sizeArr) != cudaSuccess) {
  if (cudaMalloc(&d_a, sizeArr) != cudaSuccess) {
    cout << "ERR: Unable to cudaMalloc" << cudaSuccess << endl;
    return 1;
  }
  if (cudaMalloc(&d_b, sizeArr) !=cudaSuccess) {
    cout << "ERR: Unable to cudaMalloc" << cudaSuccess << endl;
    cudaFree(d_a);
    return 1;
  }
  if (cudaMalloc(&d_c, sizeArr) != cudaSuccess) {
    cout <<  "ERR: Unable to cudaMalloc" << cudaSuccess << endl;
    cudaFree(d_a);
    cudaFree(d_b);
    return 1;
  }

  // Setup input values
  for (int i=0;i<count;i++) {
    a[i]=0;
    b[i]=count-i;
  }
  for (int i=0;i<3;i++) {
    cout << a[i] << "   " << b[i] <<endl;
  }
  // Copy inputs to device
  cudaMemcpy(d_a, &a, sizeArr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeArr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeArr, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU
  dim3 threadsPerBlock(32,320);
  cout << threadsPerBlock << endl;
  CUDAshowid<<<count/1024+1,threadsPerBlock>>>(d_a, d_b, d_c, count);
  cout << "Last Error Is = " << cudaGetLastError() << endl;
  // Copy result back to host
  cudaMemcpy(&a, d_a, sizeArr, cudaMemcpyDeviceToHost);
  // Cleanup
  cudaFree(d_a); cudaFree(d_b);

  cout << "Result is=" << endl;
  /*for (int i=0;i<3;i++) {
    cout << a[i]<< endl;
  }*/
  for (int i=0; i<=count/16;i++) {
    for (int j=0;j<16;j++){
      if ( 16*i+j<count ) {cout << a[16*i+j] << " ";}
    }
    cout << endl;
  }
  return 0;
}
