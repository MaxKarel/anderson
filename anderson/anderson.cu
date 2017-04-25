#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

using namespace std;

typedef thrust::complex<double> th_complex;

const int xmax = 256;
const int tmax = 500;
double dt = 0.02;
//float dx = 2*sqrt(dt);
double scale = 2.0;
double dx = 1.0;
th_complex imag_one (0.0, 1.0);
th_complex a = -imag_one*th_complex(dt/(dx*dx),0.0);
th_complex b = 1.0+a;

__global__ void cutrid_RC_1b(th_complex *a,th_complex *b,th_complex *c,th_complex *d,th_complex *x) {
  int idx_global=blockIdx.x*blockDim.x+threadIdx.x;
  int idx=threadIdx.x;

  __shared__ th_complex asub[255];
  __shared__ th_complex bsub[255];
  __shared__ th_complex csub[255];
  __shared__ th_complex dsub[255];

  asub[idx]=a[idx_global];
  bsub[idx]=b[idx_global];
  csub[idx]=c[idx_global];
  dsub[idx]=d[idx_global];
  __syncthreads();
  //Reduction
  for(int stride=1;stride<255;stride*=2) {
    int margin_left=(idx-stride);
    int margin_right=(idx+stride);
    if(margin_left<0) margin_left=0;
    if(margin_right>=255) margin_right=254;
    th_complex tmp1 = asub[idx] / bsub[margin_left];
    th_complex tmp2 = csub[idx] / bsub[margin_right];
    th_complex tmp3 = dsub[margin_right];
    th_complex tmp4 = dsub[margin_left];
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
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]);
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]);
void transpose(th_complex arr[][xmax]);
void altCPUa(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b);
void altCPUb(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b);
void altCPUc(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b);


th_complex h_u[xmax][xmax] = {}; //alocating on heap
th_complex h_uH[xmax][xmax] = {};
th_complex h_V[xmax][xmax] = {};
th_complex h_c[xmax] = {};
th_complex h_b[xmax] = {};
th_complex h_a[xmax] = {};
th_complex h_d[xmax] = {};
th_complex h_x[xmax] = {};

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}


  float arrSize = sizeof(th_complex) * xmax * xmax;
  float vektSize = sizeof(th_complex) * xmax;

  initializeHostArrays(h_u, h_uH, h_V, h_c);

  //////////////////////////////////////////////
  ////  CUDA                                ////
  //////////////////////////////////////////////
  /*
  ofstream r_foutCU("and_stdDev_rCU.dat");

  th_complex *d_u, *d_uH, *d_V, *d_c;
  th_complex *d_b, *d_d, *d_a, *d_x;
  cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);
  const cuDoubleComplex* _alpha = &alpha;
  const cuDoubleComplex* _beta = &beta;
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaMalloc(&d_u, arrSize);
  cudaMalloc(&d_uH, arrSize);
  cudaMalloc(&d_V, arrSize);
  cudaMalloc(&d_c, vektSize);
  cudaMalloc(&d_a, vektSize);
  cudaMalloc(&d_b, vektSize);
  cudaMalloc(&d_d, vektSize);
  cudaMalloc(&d_x, vektSize);
  checkCudaErrors(cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_uH, h_uH, arrSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_V, h_V, arrSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c, h_c, vektSize, cudaMemcpyHostToDevice));
  dim3 dimBlock(32,32);
  dim3 dimGrid(xmax / 32 + 1, xmax / 32 + 1);

  cuDoubleComplex* d_uc = reinterpret_cast<cuDoubleComplex* >(d_u);
  cuDoubleComplex* d_uHc = reinterpret_cast<cuDoubleComplex* >(d_uH);
  cuDoubleComplex* d_Vc = reinterpret_cast<cuDoubleComplex* >(d_V);

  for(int i = 0; i < xmax; i++) {
    h_c[i] = -a/2.0;
    h_a[i] = -a/2.0;
  }
  cudaMemcpy(d_a, h_a, vektSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, vektSize, cudaMemcpyHostToDevice);
  for (int t = 0; t< tmax; t++) {
    cudaMemcpy(h_uH, d_u, arrSize, cudaMemcpyDeviceToHost);
    for(int i = 1 ; i < xmax ; i++) {
      for(int j = 0 ; j < xmax ; j++) {
        h_b[j] = b - h_V[i][j];
        h_d[j] = (1.0-a)*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
      }
      cudaMemcpy(d_b, h_b, vektSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_d, h_d, vektSize, cudaMemcpyHostToDevice);
      cutrid_RC_1b<<<xmax,xmax>>>(d_a, d_b, d_c, d_d, d_x);
      cudaMemcpy(h_x, d_x, vektSize, cudaMemcpyDeviceToHost);
      for(int j = 0 ; j < xmax ; j++) {
        h_u[i][j] = h_x[j];
      }
    }
    cudaMemcpy(d_u, h_u, arrSize, cudaMemcpyHostToDevice);
    transpose(h_V);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_Vc, xmax,
                _beta, d_Vc, xmax,
                d_uHc, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_uc, xmax,
                _beta, d_uc, xmax,
                d_uHc, xmax);


    // cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
    //             _alpha, d_Vc, xmax,
    //             _beta, d_Vc, xmax,
    //             d_uHc, xmax);
    // cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    // cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
    //             _alpha, d_uc, xmax,
    //             _beta, d_uc, xmax,
    //             d_uHc, xmax);

    if (t%100==0) {
      cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      stdDev_r(r_foutCU,t,h_u);
    }
  }

  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  printResult(h_u, h_uH, h_V, h_c);
  cudaFree(d_u);
  cudaFree(d_V);
  cudaFree(d_uH);
  cudaFree(d_c);
*/
  //////////////////////////////////////////////
  ////  CPU                                 ////
  //////////////////////////////////////////////
  ofstream r_foutCPU("and_stdDev_rCPU.dat");
  for (int t = 0; t< tmax; t++) {
    altCPUa(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    altCPUa(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    if (t%100==0) {
      stdDev_r(r_foutCPU,t,h_u);
    }
  }
  printResult(h_u, h_uH, h_V, h_c);

  return 0;
}

void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j] =th_complex(0.0, 0.0);
      h_uH[i][j]=th_complex(0.0, 0.0);
      h_V[i][j] =2*scale*(double)(rand()%10000/10000.0-0.5);
      h_V[i][j] *= dt/imag_one;
      //h_V[i][j] = 0;
    }
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = th_complex(1.0, 0);
  h_uH[xmax/2][xmax/2] = th_complex(1.0, 0);
}
void printInitialVariables(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;
}
void printResult(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax],th_complex h_c[]) {
  printInitialVariables(h_u, h_uH, h_V, h_c);
  ofstream fout("data.dat");
  double sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      double probability = h_u[i][j].real()*h_u[i][j].real();
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
//Potencial na lavej
void altCPUa(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b) {

  th_complex mod_rs[xmax];  //modified right side

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
    if(h_u[i][0].real() != 0.0) {cout << "warning h_u[i][0] ==" << fixed << h_u[i][0].real() << endl;}
    if(h_u[i][0].imag() != 0.0) {cout << "warning h_u[i][0] ==" << fixed << h_u[i][0].imag() << endl;}

    if(h_u[i][xmax-1].real() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}
    if(h_u[i][xmax-1].imag() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}

    if(h_u[0][i].real() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}
    if(h_u[0][i].imag() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}

    if(h_u[xmax-1][i].real() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
    if(h_u[xmax-1][i].imag() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
  }
}
//potencial na pravej
void altCPUb(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b) {

  th_complex mod_rs[xmax];  //modified right side

  for(int i = 0 ; i < xmax ; i++) {
    for (int j = 0 ; j < xmax ; j++) {
      h_uH[i][j] = h_u[i][j]; //This is preserved state in time = t
    }
  }
  //calculate h_c
  for(int j = 0 ; j < xmax ; j++) {
    h_c[j] = -a/2.0;	//spodna diagonala v matici, je pri \psi(t-\Delta)
  }
  //modify h_c
  h_c[0] /= b;	//delime strednou diagonalou
  for(int j = 1 ; j < xmax ; j++) {
    h_c[j] /= (b) + a/2.0*h_c[j-1];	//spodna diagonala v matici je -a/2 preto +
  }

  for(int i = 1; i<xmax-1; i++) {

    mod_rs[0]  = h_V[i][0] + (1.0-a)*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
    mod_rs[0] /= b;
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di = h_V[i][j] + (1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b)+a/2.0*h_c[j-1]);
    }
    h_u[i][xmax-1]=0; //mod_rs[j];
    for(int j=xmax-2; j>0; j--) {
      h_u[i][j]=mod_rs[j]-h_c[j]*h_u[i][j+1];
    }
  }
  cout.precision(17);
  //Kontrola ci okrajove body v mriezke su = 0
  for(int i = 0 ; i < xmax ; i++) {
    if(h_u[i][0].real() != 0.0) {cout << "warning h_u[i][0] ==" << fixed << h_u[i][0].real() << endl;}
    if(h_u[i][0].imag() != 0.0) {cout << "warning h_u[i][0] ==" << fixed << h_u[i][0].imag() << endl;}

    if(h_u[i][xmax-1].real() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}
    if(h_u[i][xmax-1].imag() != 0.0) {cout << "warning h_u[i][xmax-1] ==" << h_u[i][xmax-1] << endl;}

    if(h_u[0][i].real() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}
    if(h_u[0][i].imag() != 0.0) {cout << "warning h_u[0][i] ==" << h_u[0][i] << endl;}

    if(h_u[xmax-1][i].real() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
    if(h_u[xmax-1][i].imag() != 0.0) {cout << "warning h_u[xmax-1][i] ==" << h_u[xmax-1][i] << endl;}
  }
}
//potencial na oboch
void altCPUc(th_complex h_u[][xmax], th_complex h_V[][xmax], th_complex h_c[],
            int xmax, th_complex a, th_complex b) {

  th_complex mod_rs[xmax];  //modified right side

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
    h_c[0] /= b - 0.5*h_V[i][0];	//delime strednou diagonalou
    for(int j = 1 ; j < xmax ; j++) {
      h_c[j] /= (b - 0.5*h_V[i][j]) + a/2.0*h_c[j-1];	//spodna diagonala v matici je -a/2 preto +
    }

    mod_rs[0]  = 0.5*h_V[i][0] + (1.0-a)*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
    mod_rs[0] /= b - 0.5*h_V[i][0];
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di = 0.5*h_V[i][j] + (1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b - 0.5*h_V[i][j])+a/2.0*h_c[j-1]);
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
