#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <thrust/complex.h>

using namespace std;

const int xmax = 101;
int tmax = 2000;
float tSec = 0.1;
float D = 0.5;
float l = 1;
float dx = l / (float)xmax;
float dt = tSec / (float)tmax;
float a = (D*dt)/(dx*dx);
//float b = 1+2*a;
float b = 1+a;

__global__ void explCU(thrust::complex<float>* d_u,
                       thrust::complex<float>* d_uH, int xmax, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < xmax)&(j<xmax-1)&(j>0)) {
    d_u[i*xmax+j] = a*d_uH[i*xmax+j-1]+(1.0f-2.0f*a)*d_uH[i*xmax+j]+a*d_uH[i*xmax+j+1];
  }
}

__global__ void implCU(thrust::complex<float>* d_u,
                       thrust::complex<float>* d_c, int xmax, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<xmax) {
    d_u[i*xmax + 0] = d_u[i*xmax + 0]/b;
    for(int j=1; j < xmax-1; j++) {
      d_u[i*xmax + j] = (d_u[i*xmax + j]+a*d_u[i*xmax + j-1])/(b+a*d_c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      d_u[i*xmax + j]=d_u[i*xmax + j]-d_c[j]*d_u[i*xmax + j+1];
    }
  }
}

__global__ void altCU(thrust::complex<float>* d_u,
                      thrust::complex<float>* d_uH,
                      thrust::complex<float>* d_c,
                      int xmax, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    d_uH[i*xmax + 0] = (1-a)*d_u[i*xmax + 0] + a*(d_u[(i-1)*xmax + 0]+d_u[(i+1)*xmax + 0])/2.0f;
    d_uH[i*xmax + 0] /=b;

    thrust::complex<float> di;
    for(int j=1; j < xmax-1; j++) {
      di = (1-a)*d_u[i*xmax + j] + a*(d_u[(i-1)*xmax + j]+d_u[(i+1)*xmax + j])/2.0f;
      d_uH[i*xmax + j] = (di+a/2.0f*d_uH[i*xmax + j-1])/((1+a)+a/2.0f*d_c[j-1]);
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
  if ((i < xmax)&(j<xmax)) {
    d_uH[i*xmax+j] = d_u[j*xmax+i];
  }
}
void printInitialVariables(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]);
void printResult(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]);
void initializeHostArrays(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]);
void stdDev_r(ofstream& r, float t, thrust::complex<float> u[][xmax]);


int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  thrust::complex<float> h_uH[xmax][xmax];
  thrust::complex<float> h_u[xmax][xmax];
  thrust::complex<float> h_c[xmax];
  thrust::complex<float> *d_u, *d_uH, *d_c;

  float arrSize = sizeof(thrust::complex<float>) * xmax * xmax;
  float vektSize = sizeof(thrust::complex<float>) * xmax;

  initializeHostArrays(h_u, h_uH, h_c);
  cudaMalloc(&d_u, arrSize);
  cudaMalloc(&d_uH, arrSize);
  cudaMalloc(&d_c, vektSize);
  cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_uH, h_uH, arrSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, vektSize, cudaMemcpyHostToDevice);

  ofstream r_fout("stdDev_r.dat");
  dim3 dimBlock(32,32);
  dim3 dimGrid(xmax / 32 + 1, xmax / 32 + 1);
  for (int t = 0; t< tmax; t++) {
    altCU<<<1,xmax>>>(d_u, d_uH, d_c, xmax, a, b);
    transposeCU<<<dimGrid,dimBlock>>>(d_u, d_uH, xmax);
    cudaMemcpy(d_u, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    altCU<<<1,xmax>>>(d_u, d_uH, d_c, xmax, a, b);
    transposeCU<<<dimGrid,dimBlock>>>(d_u, d_uH, xmax);
    cudaMemcpy(d_u, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    if (t%100==0) {
      cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      stdDev_r(r_fout,t,h_u);
    }
  }
  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_uH, d_uH, arrSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, vektSize, cudaMemcpyDeviceToHost);
  printResult(h_uH, h_u, h_c);
  cudaFree(d_u);
  cudaFree(d_uH);
  cudaFree(d_c);
  return 0;
}

void initializeHostArrays(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_uH[i][j]=thrust::complex<float>(0.0f, 0.0f); h_u[i][j]=thrust::complex<float>(0.0f, 0.0f);
    }
    h_c[i]=-a/2.0f;
  }
  h_c[0] /= 1+a;
  for (int i = 1; i<xmax; i++) {
    h_c[i] = h_c[i]/((1+a) + a*h_c[i-1]/2.0f);
  }
  h_u[xmax/2][xmax/2] = 10000;
  h_uH[xmax/2][xmax/2] = thrust::complex<float>(10000.0f, 0.0f);
}
void printInitialVariables(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]) {
  cout << " dx==" << dx <<endl<< " dt==" << dt << endl << " a==" << a << endl;
  cout << "b==" << b << endl;
  cout << "ARR C  " << endl << endl;
  for (int i = 0; i< xmax/16 + 1; i++) {
    for(int j= 0; j<16 ; j++) {
      if ((i*32 + j)<xmax){cout<<h_c[i*32 + j] << " ";}
    }
    cout << endl;
  }
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j< xmax; j++) {
      //cout<<h_u[i][j] << " ";
    }
    //cout <<endl;
  }
}
void printResult(thrust::complex<float> h_uH[][xmax], thrust::complex<float> h_u[][xmax], thrust::complex<float> h_c[]) {
  ofstream fout("data.dat");
  ofstream fout2("real.dat");
  ofstream fout3("real_dev.dat");
  float konst = 1/296.4f;
  float citatel, menovatel,value,sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      fout << i << " " << j << " " <<  h_u[i][j].real() << endl;
      citatel = pow((float)(i-xmax/2)/xmax,2) + pow((float)(j-xmax/2)/xmax,2);
      menovatel = 2.0f * D * tSec;
      value = (konst / tSec) * exp(-citatel / menovatel);
      sum+=value;
      fout2 << i <<" "<< j <<" "<< value << endl;
    }
  }
  fout.close();
  for(int t = 0; t<tmax; t+=100) {
    fout3 << (float)t/tmax*tSec << " " << 2*D*t/tmax*tSec << endl;
  }
  fout3.close();
  cout << h_u[xmax/2][xmax/2] << endl;
  sum = 0;
  for(int i=0; i<xmax; i++){
    for(int j = 0; j<xmax; j++) {
      sum += h_u[i][j].real();
    }
  }
  cout << "sum == " << sum << endl;
  printInitialVariables(h_uH, h_u, h_c);
}
void stdDev_r(ofstream& r, float t, thrust::complex<float> u[][xmax]) {
  thrust::complex<float> sum = 0;
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j < xmax; j++) {
      sum += (float)(pow((float)(i-xmax/2)/xmax*l,2) + pow((float)(j-xmax/2)/xmax*l,2))*u[i][j]/10000.0f;
    }
  }
  r << tSec*t/tmax << " " << sum.real() << endl;
}
