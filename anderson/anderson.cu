#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "cusolverDn.h"
#include "helper_cuda.h"

using namespace std;

typedef thrust::complex<double> th_complex;

const int xmax = 201;
const int tmax = 150000;
double dt = 0.002;
//float dx = 2*sqrt(dt);
double scale = 2.0;
double dx = 1.0;
th_complex imag_one (0.0, 1.0);
th_complex a = -imag_one*th_complex(dt/(dx*dx),0.0);
th_complex b = 1.0+a;

__global__ void altCU(th_complex* d_u, th_complex* d_uH, th_complex* d_V,
            int xmax, th_complex a, th_complex b) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    th_complex mod_rs[201];  //modified right side
    th_complex c[201];  //super-diagonal vector
    //calculate h_c
    for(int j = 0 ; j < xmax ; j++) {
      c[j] = -a/2.0;	//spodna diagonala v matici, je pri \psi(t-\Delta)
    }
    //modify h_c
    c[0] /= b - d_V[i*xmax + 0];	//delime strednou diagonalou
    for(int j = 1 ; j < xmax ; j++) {
      c[j] /= (b - d_V[i*xmax + j]) + a/2.0*c[j-1];	//spodna diagonala v matici je -a/2 preto +
    }

    mod_rs[0]  = (1.0-a)*d_uH[i*xmax + 0] + a/2.0*(d_uH[(i-1)*xmax + 0]+d_uH[(i+1)*xmax + 0]);
    mod_rs[0] /= b - d_V[i*xmax + 0];
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di  = (1.0-a)*d_uH[i*xmax + j] + a/2.0*(d_uH[(i-1)*xmax + j]+d_uH[(i+1)*xmax + j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b - d_V[i*xmax + j])+a/2.0*c[j-1]);
    }
    d_u[i*xmax + xmax-1]=0; //mod_rs[j];
    for(int j=xmax-2; j>0; j--) {
      d_u[i*xmax + j]=mod_rs[j]-c[j]*d_u[i*xmax + j+1];

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
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]);
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]);
void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]);
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]);
void transpose(th_complex arr[][xmax]);
void altCPU(th_complex h_u[][xmax], th_complex h_V[][xmax],
            int xmax, th_complex a, th_complex b);

  th_complex h_u[xmax][xmax] = {}; //alocating on heap
  th_complex h_uH[xmax][xmax] = {};
  th_complex h_V[xmax][xmax] = {};
  th_complex h_B[xmax][xmax] = {};
  th_complex h_d[xmax] = {};

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}


  float arrSize = sizeof(th_complex) * xmax * xmax;
  float vektSize = sizeof(th_complex) * xmax;

  initializeHostArrays(h_u, h_uH, h_V);

  //////////////////////////////////////////////
  ////  CUDA                                ////
  //////////////////////////////////////////////
  th_complex *d_u, *d_uH, *d_V, *d_B, *d_d;
  cudaMalloc(&d_u, arrSize);
  cudaMalloc(&d_uH, arrSize);
  cudaMalloc(&d_V, arrSize);
  cudaMalloc(&d_B, arrSize);
  cudaMalloc(&d_d, vektSize);
  cuDoubleComplex* d_uc = reinterpret_cast<cuDoubleComplex* >(d_u);
  cuDoubleComplex* d_uHc = reinterpret_cast<cuDoubleComplex* >(d_uH);
  cuDoubleComplex* d_Vc = reinterpret_cast<cuDoubleComplex* >(d_V);
  cuDoubleComplex* d_Bc = reinterpret_cast<cuDoubleComplex* >(d_B);
  cuDoubleComplex* d_dc = reinterpret_cast<cuDoubleComplex* >(d_d);

  cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
  const cuDoubleComplex* _alpha = &alpha;
  const cuDoubleComplex* _beta = &beta;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cusolverStatus_t status;
  cusolverDnHandle_t handle_solver;
  cusolverDnCreate(&handle_solver);


  ofstream r_fout("and_stdDev_rCU.dat");

  checkCudaErrors(cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_uH, h_uH, arrSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_V, h_V, arrSize, cudaMemcpyHostToDevice));
  dim3 dimBlock(32,32);
  dim3 dimGrid(xmax / 32 + 1, xmax / 32 + 1);

  cuDoubleComplex* d_work;
  int work_size = 0;
  int info = 0;
  int* d_info;
  int* d_Ipiv;
  cudaMalloc(&d_Ipiv,(xmax)*sizeof(int));
  cudaMalloc(&d_info,sizeof(int));
  cudaMalloc(&d_work,work_size*sizeof(cuDoubleComplex));
  cusolverDnZgetrf_bufferSize(handle_solver,xmax,xmax,d_Bc,xmax,&work_size);
  cout.precision(25);
  for(int m = 0 ; m < xmax ; m++) {
    for(int n = 0 ; n < xmax ; n++) {
      h_B[m][n] = 0.0;
    }
  }
  for (int t = 0; t < tmax; t++) {
    /*for(int i = 0; i < xmax ; i++) {
      h_B[0][0] = b - h_V[i][0];
      h_B[0][1] = -a/2.0;
      for(int k = 1 ; k < xmax -1 ; k++) {
        h_B[k][k-1] = -a/2.0;
        h_B[k][k] = b - h_V[i][k];
        h_B[k][k+1] = -a/2.0;
      }
      h_B[xmax-1][xmax-2] = -a/2.0;
      h_B[xmax-1][xmax-1] = b - h_V[i][xmax-1];
      cudaMemcpy(d_B, h_B, arrSize, cudaMemcpyHostToDevice);
      for(int j = 0 ; j < xmax ; j++) {
        h_d[j] = (1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
      }
      cudaMemcpy(d_d, h_d, vektSize, cudaMemcpyHostToDevice);

      cusolverDnZgetrf(handle_solver,xmax,xmax,d_Bc,xmax,d_work,d_Ipiv,d_info);
      cudaMemcpy(&info, d_info,sizeof(int),cudaMemcpyDeviceToHost);
    	if(info!=0){
    			cout << info << endl;
    		}
      status = cusolverDnZgetrs(handle_solver,CUBLAS_OP_N ,xmax,1,d_Bc,xmax,d_Ipiv,d_dc,xmax,d_info);
    	if(status!=CUSOLVER_STATUS_SUCCESS){
    		cout << status << endl;
    	}
      cudaMemcpy(d_u+i ,d_d,vektSize,cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
    }

    transpose(h_V);*/
    altCU<<<1,xmax>>>(d_u, d_uH, d_V, xmax, a, b);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_Vc, xmax,
                _beta, d_Vc, xmax,
                d_uHc, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_uc, xmax,
                _beta, d_uc, xmax,
                d_uHc, xmax);

    altCU<<<1,xmax>>>(d_u, d_uH, d_V, xmax, a, b);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_Vc, xmax,
                _beta, d_Vc, xmax,
                d_uHc, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_uc, xmax,
                _beta, d_uc, xmax,
                d_uHc, xmax);
    //cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    if (t%100==0) {
      cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      stdDev_r(r_fout,t,h_u);
    }
  }

  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  printResult(h_u, h_uH, h_V);
  cudaFree(d_u);
  cudaFree(d_V);
  cudaFree(d_uH);

  //////////////////////////////////////////////
  ////  CPU                                 ////
  //////////////////////////////////////////////
  /*ofstream r_fout("std.cpp");
  for (int t = 0; t< tmax; t++) {
    altCPU(h_u, h_V, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    altCPU(h_u, h_V, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    if (t%10==0) {
      stdDev_r(r_fout,t,h_u);
    }
  }
  printResult(h_u, h_uH, h_V, h_c);
  */

  return 0;
}

void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j] =th_complex(0.0, 0.0);
      h_uH[i][j]=th_complex(0.0, 0.0);
      h_V[i][j] =th_complex(2*scale*(float)(rand()%10000/10000.0-0.5), 0.0);
      h_V[i][j] *= th_complex(dt,0.0)/imag_one;
      //h_V[i][j] = 0;
    }
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = th_complex(1.0, 0);
  h_uH[xmax/2][xmax/2] = th_complex(1.0, 0);
}
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;
}
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax]) {
  printInitialVariables(h_u, h_uH, h_V);
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
      sum += (float)(pow((float)(i-xmax/2),2) + pow((float)(j-xmax/2),2))*(u[i][j].real()*u[i][j].real() + u[i][j].imag()*u[i][j].imag());
    }
  }
  r << t/tmax << " " << sum.real() << endl;
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

void altCPU(th_complex h_u[][xmax], th_complex h_V[][xmax],
            int xmax, th_complex a, th_complex b) {

  th_complex mod_rs[xmax];  //modified right side
  th_complex h_c[xmax];
  for(int i = 0 ; i < xmax ; i++) {
    for (int j = 0 ; j < xmax ; j++) {
      h_uH[i][j] = h_u[i][j]; //This is preserved state in time = t
    }
  }

  for(int i = 1; i<xmax-1; i++) {
    //calculate h_c
    for(int j = 0 ; j < xmax ; j++) {
      h_c[j] = -a/2.0;	//spodna diagonala v matici, je pri \psi(t-\Delta)
    }
    //modify h_c
    h_c[0] /= b - h_V[i][0];	//delime strednou diagonalou
    for(int j = 1 ; j < xmax ; j++) {
      h_c[j] /= (b - h_V[i][j]) + a/2.0*h_c[j-1];	//spodna diagonala v matici je -a/2 preto +
    }

    mod_rs[0]  = (1.0-a)*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
    mod_rs[0] /= b - h_V[i][0];
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di  = (1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b - h_V[i][j])+a/2.0*h_c[j-1]);
    }
    h_u[i][xmax-1]=0; //mod_rs[j];
    for(int j=xmax-2; j>0; j--) {
      h_u[i][j]=mod_rs[j]-h_c[j]*h_u[i][j+1];
    }
  }
  cout.precision(17);
  //Kontrola ci okrajove body v mriezke su = 0
  for(int i = 0 ; i < xmax ; i++) {
    if(h_u[i][0].real() != 0.0) {cout << setprecision(10) << "warning h_u[i][0] ==" << fixed << h_u[i][0].real() << endl;}
    if(h_u[i][0].imag() != 0.0) {cout << setprecision(10) << "warning h_u[i][0] ==" << fixed << h_u[i][0].imag() << endl;}

    if(h_u[i][xmax-1].real() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}
    if(h_u[i][xmax-1].imag() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}

    if(h_u[0][i].real() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}
    if(h_u[0][i].imag() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}

    if(h_u[xmax-1][i].real() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
    if(h_u[xmax-1][i].imag() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
  }
}
