#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <fstream>
#include <cstdio>
#include <ctime>

#include <thrust/complex.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

#define LATTICE_SIZE 512

using namespace std;

typedef thrust::complex<double> th_complex;

const int xmax = LATTICE_SIZE;
const int tmax = 50;
double dt = 0.02;
//float dx = 2*sqrt(dt);
double scale = 2.0;
double dx = 1.0;
th_complex imag_one (0.0, 1.0);
th_complex a = -imag_one*th_complex(dt/(dx*dx),0.0);
th_complex b = 1.0+a;

__global__ void altCUc(th_complex* d_u, th_complex* d_uH, th_complex* d_V,
            int xmax, th_complex a, th_complex b) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    th_complex mod_rs[LATTICE_SIZE];  //modified right side
    th_complex d_c[LATTICE_SIZE];
    //calculate h_c
    for(int j = 0 ; j < xmax ; j++) {
      d_c[j] = -a/2.0;	//spodna diagonala v matici, je pri \psi(t-\Delta)
    }
    //modify h_c
    d_c[0] /= b - 0.5*d_V[i*xmax+0];	//delime strednou diagonalou
    for(int j = 1 ; j < xmax ; j++) {
      d_c[j] /= (b - 0.5*d_V[i*xmax+j]) + a/2.0*d_c[j-1];	//spodna diagonala v matici je -a/2 preto +
    }

    mod_rs[0]  =  (0.5*d_V[i*xmax+0] + 1.0-a)*d_uH[i*xmax+0] + a/2.0*(d_uH[(i-1)*xmax+0]+d_uH[(i+1)*xmax+0]);
    mod_rs[0] /= b - 0.5*d_V[i*xmax+0];
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di = (0.5*d_V[i*xmax+j] + 1.0-a)*d_uH[i*xmax+j] + a/2.0*(d_uH[(i-1)*xmax+j]+d_uH[(i+1)*xmax+j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b - 0.5*d_V[i*xmax+j])+a/2.0*d_c[j-1]);
    }
    d_u[i*xmax+xmax-1]=0; //mod_rs[j];
    for(int j=xmax-2; j>0; j--) {
      d_u[i*xmax+j]=mod_rs[j]-d_c[j]*d_u[i*xmax+j+1];
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
void printResult(th_complex h_u[][xmax],ofstream& data_file, ofstream& time_file, float elapsed_time, string platform);
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

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  float arrSize = sizeof(th_complex) * xmax * xmax;
  float vektSize = sizeof(th_complex) * xmax;
  //Create folder and output files
  string DIR = "results_anderson/";
  string DATA_GPU = "data_gpu.dat";
  string DATA_CPU = "data_cpu.dat";
  string DEV_GPU = "dev_gpu.dat";
  string DEV_CPU = "dev_cpu.dat";
  string DATA_TIME = "time.dat";
  const char* MKDIR = "mkdir -p ";

  string size_of_lattice = to_string(xmax);
  string scale_of_lattice = to_string((int)scale);
  string number_of_iterations = to_string(tmax);
  DIR += "x_"+size_of_lattice+"_s_"+scale_of_lattice+"_t_"+number_of_iterations+"/";
  const char* PATH = DIR.c_str();

  int LEN_MKDIR = strlen(MKDIR) + strlen(PATH);
  char* CMD_MKDIR = (char*)malloc(LEN_MKDIR*sizeof(char));
  sprintf(CMD_MKDIR, "%s%s", MKDIR, PATH);
  cout << "Saving output in:  " << PATH << endl;
  system(CMD_MKDIR);

  DATA_GPU = DIR + DATA_GPU;
  ofstream out_data_gpu(DATA_GPU);
  DATA_CPU = DIR + DATA_CPU;
  ofstream out_data_cpu(DATA_CPU);
  DEV_GPU = DIR + DEV_GPU;
  ofstream out_dev_gpu(DEV_GPU);
  DEV_CPU = DIR + DEV_CPU;
  ofstream out_dev_cpu(DEV_CPU);
  DATA_TIME = DIR + DATA_TIME;
  ofstream out_time(DATA_TIME);

  //////////////////////////////////////////////
  ////  CUDA                                ////
  //////////////////////////////////////////////

  cudaEvent_t start_gpu, end_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventRecord(start_gpu);

  initializeHostArrays(h_u, h_uH, h_V, h_c);

  th_complex *d_u, *d_uH, *d_V, *d_c;
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
  }

  for (int t = 0; t< tmax; t++) {
    altCUc<<<1,xmax>>>(d_u,d_uH,d_V,xmax,a,b);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_Vc, xmax,
                _beta, d_Vc, xmax,
                d_uHc, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_uc, xmax,
                _beta, d_uc, xmax,
                d_uHc, xmax);

    altCUc<<<1,xmax>>>(d_u,d_uH,d_V,xmax,a,b);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_Vc, xmax,
                _beta, d_Vc, xmax,
                d_uHc, xmax);
    cudaMemcpy(d_V, d_uH, arrSize, cudaMemcpyDeviceToDevice);
    cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, xmax, xmax,
                _alpha, d_uc, xmax,
                _beta, d_uc, xmax,
                d_uHc, xmax);

    if (t%100==0) {
      cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
      stdDev_r(out_dev_gpu,t,h_u);
    }
  }

  cudaMemcpy(h_u, d_u, arrSize, cudaMemcpyDeviceToHost);
  cudaFree(d_u);
  cudaFree(d_V);
  cudaFree(d_uH);
  cudaFree(d_c);

  cudaEventRecord(end_gpu);
  cudaEventSynchronize(end_gpu);
  float time_gpu = 0; //miliseconds
  cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);
  time_gpu /= 1000;
  cout << "TIME GPU:   " << time_gpu << " s" << endl;
  printResult(h_u, out_data_gpu, out_time, time_gpu, "GPU");

  //////////////////////////////////////////////
  ////  CPU                                 ////
  //////////////////////////////////////////////

  clock_t start_cpu = clock();
  for(int i = 0 ; i < xmax ; i++) {
    for(int j = 0 ; j < xmax ; j++) {
      h_u[i][j] = 0;
    }
  }
  h_u[xmax/2][xmax/2] = 1.0;
  for (int t = 0; t< tmax; t++) {
    altCPUc(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    altCPUc(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    if (t%10==0) {
      stdDev_r(out_dev_cpu,t,h_u);
    }
  }

  clock_t end_cpu = clock();
  double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  cout << "TIME CPU:   " << time_cpu << " s" << endl;
  printResult(h_u, out_data_cpu, out_time, time_cpu, "CPU");

  return 0;
}

void initializeHostArrays(th_complex h_u[][xmax], th_complex h_uH[][xmax], th_complex h_V[][xmax], th_complex h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j] =th_complex(0.0, 0.0);
      h_uH[i][j]=th_complex(0.0, 0.0);
      h_V[i][j] =2*scale*(double)(rand()%10000/10000.0-0.5);
      h_V[i][j] *= dt/imag_one;
    }
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = th_complex(1.0, 0);
  h_uH[xmax/2][xmax/2] = th_complex(1.0, 0);
}
void printResult(th_complex h_u[][xmax], ofstream& data_file, ofstream& time_file, float elapsed_time, string platform) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;

  double sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      double probability = h_u[i][j].real()*h_u[i][j].real();
      probability += h_u[i][j].imag()*h_u[i][j].imag();
      data_file << i << " " << j << " " <<  probability << " " << endl;
      sum += probability;
    }
  }
  cout << h_u[xmax/2][xmax/2] << endl;
  cout << "sum == " << sum << endl;

  time_file << "***************" << endl;
  time_file << "Runtime for " << platform << " was: " << elapsed_time << " s" << endl;
  time_file << "Probability sum was: " << sum << endl;
  time_file << "Probability in middle (" << xmax << "," << xmax << ") was" << h_u[xmax/2][xmax/2] << endl;

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

    mod_rs[0]  =  (1.0-a+h_V[i][0])*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
    mod_rs[0] /= b;
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di = (1.0-a+h_V[i][0])*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
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

    mod_rs[0]  =  (0.5*h_V[i][0] + 1.0-a)*h_uH[i][0] + a/2.0*(h_uH[i-1][0]+h_uH[i+1][0]);
    mod_rs[0] /= b - 0.5*h_V[i][0];
    th_complex di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di = (0.5*h_V[i][j] + 1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
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
