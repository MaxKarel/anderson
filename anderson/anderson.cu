#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <thrust/complex.h>

using namespace std;

const int xmax = 101;
float dt = 1/100.0f;
//float dx = 2*sqrt(dt);
float dx = 1;
thrust::complex<float> imag_one = thrust::complex<float> (0.0f, 1.0f);
thrust::complex<float> a = -imag_one*dt/(dx*dx);
thrust::complex<float> b = 1.0f+a;

__global__ void altCU(thrust::complex<float>* d_u,
                      thrust::complex<float>* d_uH,
                      thrust::complex<float>* d_c,
                      thrust::complex<float>* d_V,
                      int xmax,
                      thrust::complex<float> a,
                      thrust::complex<float> b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    d_uH[i*xmax + 0]  = (1.0f-a)*d_u[i*xmax + 0] + a/2.0f*(d_u[(i-1)*xmax + 0]+d_u[(i+1)*xmax + 0]);
    d_uH[i*xmax + 0] += d_V[i*xmax + 0];  //d_V je uz vynasobene konstantami v init funkcii
    d_uH[i*xmax + 0] /= b;
    thrust::complex<float> di;
    for(int j=1; j < xmax-1; j++) {
      di  = d_V[i*xmax + j];
      di += (1.0f-a)*d_u[i*xmax + j] + a/2.0f*(d_u[(i-1)*xmax + j]+d_u[(i+1)*xmax + j]);

      d_uH[i*xmax + j] = (di+a/2.0f*d_uH[i*xmax + j-1])/(b+a/2.0f*d_c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      d_u[i*xmax + j]=d_uH[i*xmax + j]-d_c[j]*d_u[i*xmax + j+1];
    }
  }
}

__global__ void transposeCU(thrust::complex<float>* d_u,
                            thrust::complex<float>* d_uH,int xmax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < xmax)&(j < xmax)) {
    d_uH[i*xmax+j] = d_u[j*xmax+i];
  }
}
void printInitialVariables(thrust::complex<float> h_u[][xmax],
                           thrust::complex<float> h_uH[][xmax],
                           thrust::complex<float> h_V[][xmax],
                           thrust::complex<float> h_c[]);
void printResult(thrust::complex<float> h_u[][xmax],
                 thrust::complex<float> h_uH[][xmax],
                 thrust::complex<float> h_V[][xmax],
                 thrust::complex<float> h_c[]);
void initializeHostArrays(thrust::complex<float> h_u[][xmax],
                          thrust::complex<float> h_uH[][xmax],
                          thrust::complex<float> h_V[][xmax],
                          thrust::complex<float> h_c[]);

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  thrust::complex<float> h_uH[xmax][xmax];
  thrust::complex<float> h_u[xmax][xmax];
  thrust::complex<float> h_V[xmax][xmax];
  thrust::complex<float> h_c[xmax];
  thrust::complex<float> *d_u, *d_uH, *d_V, *d_c;

  float arrSize = sizeof(thrust::complex<float>) * xmax * xmax;
  float vektSize = sizeof(thrust::complex<float>) * xmax;
  initializeHostArrays(h_u, h_uH, h_V, h_c);
  cudaMalloc(&d_u, arrSize);
  cudaMalloc(&d_uH, arrSize);
  cudaMalloc(&d_V, arrSize);
  cudaMalloc(&d_c, vektSize);
  cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uH, h_uH, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, vektSize, cudaMemcpyHostToDevice);

  dim3 dimBlock(32,32);
  dim3 dimGrid(xmax / 32 + 1, xmax / 32 + 1);
  for (int t = 0; t< 40; t++) {
    altCU<<<dimGrid,dimBlock>>>(d_u, d_uH, d_c, d_V,xmax,a,b);
    transposeCU<<<dimGrid,dimBlock>>>(d_u, d_uH, xmax);
    cudaMemcpy(d_u, d_uH, arrSize, cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, vektSize, cudaMemcpyDeviceToHost);
  printResult(h_u, h_uH, h_V, h_c);
  cudaFree(d_u);
  cudaFree(d_uH);
  cudaFree(d_c);
  return 0;
}

void initializeHostArrays(thrust::complex<float> h_u[][xmax],
                          thrust::complex<float> h_uH[][xmax],
                          thrust::complex<float> h_V[][xmax],
                          thrust::complex<float> h_c[]) {


  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j]=thrust::complex<float>(0.0f, 0.0f);
      h_uH[i][j]=thrust::complex<float>(0.0f, 0.0f);
      //cout << (float)(random()%1000/1000.0) << "rrrr" << endl;
      h_V[i][j]=thrust::complex<float>(2*7.0f*(float)(random()%1000/1000.0-0.5f), 0.0f);
      h_V[i][j] *= dt/imag_one;
      //h_V[i][j] = 0;
    }
    h_c[i]=-a;  //nastavenie superdiagonaly v matici B
    //B*PSI(t+\delta) = A*PSI(t)
  }
  //uprava pola C na algoritmus vypoctu systemu s trojdiagonalnou maticou
  h_c[0] /= b;
  for (int i = 1; i<xmax; i++) {
    h_c[i] = h_c[i]/(b + a*h_c[i-1]);
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = thrust::complex<float>(1.0f, 0);
}
void printInitialVariables(thrust::complex<float> h_u[][xmax],
                           thrust::complex<float> h_uH[][xmax],
                           thrust::complex<float> h_V[][xmax],
                           thrust::complex<float> h_c[]) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;
  cout << "ARR h_c:" << endl;
  for (int i = 0; i< xmax/16 + 1; i++) {
    for(int j= 0; j<16 ; j++) {
      if ((i*32 + j)<xmax){cout<<h_c[i*32 + j] << " ";}
    }
    cout << endl;
  }
}
void printResult(thrust::complex<float> h_u[][xmax],
                 thrust::complex<float> h_uH[][xmax],
                 thrust::complex<float> h_V[][xmax],
                 thrust::complex<float> h_c[]) {

  printInitialVariables(h_u, h_uH, h_V, h_c);
  ofstream fout("data.dat");
  float sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      float probability = h_u[i][j].real()*h_u[i][j].real();
      probability + h_u[i][j].imag()*h_u[i][j].imag();
      fout << i << " " << j << " " <<  probability << " " << endl;
      sum += probability;
    }
  }
  fout.close();
  cout << h_u[xmax/2][xmax/2] << endl;
  cout << "sum == " << sum << endl;
}
