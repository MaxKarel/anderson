#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <thrust/complex.h>
#include <cublas_v2.h>

using namespace std;

typedef thrust::complex<float> th_complex;

const int xmax = 101;
const int tmax = 2000;
float dt = 1/100.0f;
float dx = 2*sqrt(dt);
//float dx = 1;
th_complex imag_one = th_complex (0.0f, 1.0f);
th_complex a = -imag_one*dt/(dx*dx);
th_complex b = 1.0f+a;

__global__ void altCU(th_complex* d_u, th_complex* d_uH,
                      th_complex* d_c, th_complex* d_V,
                      int xmax, th_complex a, th_complex b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    d_uH[i*xmax + 0]  = (1.0f-a)*d_u[i*xmax + 0] + a/2.0f*(d_u[(i-1)*xmax + 0]+d_u[(i+1)*xmax + 0]);
    d_uH[i*xmax + 0] += d_V[i*xmax + 0];  //d_V je uz vynasobene konstantami v init funkcii
    d_uH[i*xmax + 0] /= b;
    th_complex di;
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

__global__ void transposeCU(th_complex* d_u,
                            th_complex* d_uH,int xmax) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < xmax)&(j < xmax)) {
    d_uH[i*xmax+j] = d_u[j*xmax+i];
  }
}
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]);
void transpose(th_complex arr[][xmax]);
void altCPU(th_complex h_u[][xmax], th_complex h_uH[][xmax],
            th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b);

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  th_complex h_u[xmax][xmax];
  th_complex h_uH[xmax][xmax];
  th_complex h_V[xmax][xmax];
  th_complex h_c[xmax];
  th_complex *d_u, *d_uH, *d_V, *d_c;

  const float alpha = 1.0;
  const float beta  = 0.0;
  cublasHandle_t handle;
  cublasCreate(&handle);

  float arrSize = sizeof(th_complex) * xmax * xmax;
  float vektSize = sizeof(th_complex) * xmax;

  initializeHostArrays(h_u, h_uH, h_V, h_c);

  //////////////////////////////////////////////
  ////  CUDA                                ////
  //////////////////////////////////////////////
  /*ofstream r_fout("stdDev_rCU.dat");
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
  for (int t = 0; t< 15; t++) {
    altCU<<<1,xmax>>>(d_u, d_uH, d_c, d_V,xmax,a,b);
    transposeCU<<<dimGrid,dimBlock>>>(d_u, d_uH, xmax);
    cudaMemcpy(d_u, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    transposeCU<<<dimGrid,dimBlock>>>(d_V, d_uH, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    if (t%100==0) {
      cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      stdDev_r(r_fout,t,h_u);
    }
  }
  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, vektSize, cudaMemcpyDeviceToHost);
  printResult(h_u, h_uH, h_V, h_c);
  cudaFree(d_u);
  cudaFree(d_uH);
  cudaFree(d_c);
  */
  //////////////////////////////////////////////
  ////  CPU                                 ////
  //////////////////////////////////////////////
  for (int t = 0; t< tmax; t++) {
    altCPU(h_u, h_uH, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    altCPU(h_u, h_uH, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    if (t%100==0) {
      //cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      //stdDev_r(r_fout,t,h_u);
    }
  }
  printResult(h_u, h_uH, h_V, h_c);
  return 0;
}

void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j]=th_complex(0.0f, 0.0f);
      h_uH[i][j]=th_complex(0.0f, 0.0f);
      h_V[i][j]=th_complex(7.0f*(float)(random()%2000/2000.0-1.0f), 0.0f);
      h_V[i][j] *= dt/imag_one;
      //h_V[i][j] = 0;
    }
    h_c[i]=-a;  //nastavenie superdiagonaly v matici B
    //B*\psi(t+\delta) = A*\psi(t)
  }
  //uprava pola C na algoritmus vypoctu systemu s trojdiagonalnou maticou
  h_c[0] /= b;
  for (int i = 1; i<xmax; i++) {
    h_c[i] = h_c[i]/(b + a*h_c[i-1]);
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = th_complex(1.0f, 0);
}
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;
}
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax],th_complex h_c[]) {
  printInitialVariables(h_u, h_uH, h_V, h_c);
  ofstream fout("data.dat");
  float sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      float probability = h_u[i][j].real()*h_u[i][j].real();
      probability += h_u[i][j].imag()*h_u[i][j].imag();
      fout << i << " " << j << " " <<  probability << " " << endl;
      sum += probability;
    }
  }
  fout.close();
  cout << h_u[xmax/2][xmax/2] << endl;
  cout << "sum == " << sum << endl;
}
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]) {
  th_complex sum = 0;
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j < xmax; j++) {
      //sum += (float)(pow((float)(i-xmax/2)/xmax*l,2) + pow((float)(j-xmax/2)/xmax*l,2))*u[i][j]/10000.0f;
    }
  }
  //r << tSec*t/tmax << " " << sum.real() << endl;
}
void transpose(th_complex arr[][xmax]) {
  th_complex help;
  for(int i = 0; i < xmax; i++) {
    for(int j = i+1; j < xmax; j++) {
      help = arr[i][j];
      arr[i][j] = arr[j][i];
      arr[j][i] = help;
    }
  }
}

void altCPU(th_complex h_u[][xmax], th_complex h_uH[][xmax],
            th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b) {
  th_complex help[xmax][xmax];
  for(int i = 0 ; i < xmax ; i++) {
    for (int j = 0 ; j < xmax ; j++) {
      help[i][j] = h_u[i][j];
    }
  }
  for(int i = 1; i<xmax-1; i++) {
    h_uH[i][0] = (1.0f-a)*help[i][0] + a*(help[i-1][0]+help[i+1][0])/2.0f;
    h_uH[i][0] += h_V[i][0];
    h_uH[i][0] /=b;
    th_complex di;
    for(int j=1; j < xmax-1; j++) {
      di  = h_V[i][j];
      di += (1.0f-a)*help[i][j] + a*(help[i-1][j]+help[i+1][j])/2.0f;
      h_uH[i][j] = (di+a/2.0f*h_uH[i][j-1])/((1.0f+a)+a/2.0f*h_c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      h_u[i][j]=h_uH[i][j]-h_c[j]*h_u[i][j+1];
    }
  }
}
