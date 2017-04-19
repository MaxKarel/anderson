#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#include <thrust/complex.h>

typedef thrust::complex<float> th_complex;
using namespace std;

const int xmax = 101;
int tmax = 2000;
float tSec = 0.1;
float D = 2;
float l = 1;
float dx = l / (float)xmax;
float dt = tSec / (float)tmax;
float a = (D*dt)/(dx*dx);
//float b = 1+2*a;
float b = 1+a;
/*
__global__ void explCU(th_complex* d_u,
                       th_complex* d_uH, int xmax, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ((i < xmax)&(j<xmax-1)&(j>0)) {
    d_u[i*xmax+j] = a*d_uH[i*xmax+j-1]+(1.0f-2.0f*a)*d_uH[i*xmax+j]+a*d_uH[i*xmax+j+1];
  }
}
__global__ void implCU(th_complex* d_u,
                       th_complex* d_c, int xmax, float a, float b) {
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
__global__ void altCU(th_complex* d_u,
                      th_complex* d_uH,
                      th_complex* d_c,
                      int xmax, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i<xmax-1)&(i>0)) {
    d_uH[i*xmax + 0] = (1-a)*d_u[i*xmax + 0] + a*(d_u[(i-1)*xmax + 0]+d_u[(i+1)*xmax + 0])/2.0f;
    d_uH[i*xmax + 0] /=b;

    th_complex di;
    for(int j=1; j < xmax-1; j++) {
      di = (1-a)*d_u[i*xmax + j] + a*(d_u[(i-1)*xmax + j]+d_u[(i+1)*xmax + j])/2.0f;
      d_uH[i*xmax + j] = (di+a/2.0f*d_uH[i*xmax + j-1])/((1+a)+a/2.0f*d_c[j-1]);
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
  if ((i < xmax)&(j<xmax)) {
    d_uH[i*xmax+j] = d_u[j*xmax+i];
  }
}
*/

void printInitialVariables(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]);
void printResult(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]);
void initializeHostArrays(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]);
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]);
void transpose(th_complex arr[][xmax]);
void altCPU(th_complex h_u[][xmax], th_complex h_uH[][xmax],
            th_complex h_V[][xmax], th_complex h_c[],
            float xmax,
            th_complex a, th_complex b);
int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  th_complex h_uH[xmax][xmax];
  th_complex h_u[xmax][xmax];
  th_complex h_V[xmax][xmax];
  th_complex h_c[xmax];

  float arrSize = sizeof(th_complex) * xmax * xmax;
  float vektSize = sizeof(th_complex) * xmax;

  initializeHostArrays(h_u, h_uH, h_c);

  ofstream r_fout("stdDev_r.dat");
  for (int t = 0; t< 2; t++) {
    copy(&h_u[0][0],&h_u[0][0]+xmax*xmax,&h_uH[0][0]);
    cout << "m=" << h_uH[xmax/2][xmax/2] << endl;
    altCPU(h_u, h_uH, h_V, h_c, xmax, a, b);
    transpose(h_u);
    if (t%100==0) {
      stdDev_r(r_fout,t,h_u);
    }
  }
  printResult(h_uH, h_u, h_c);
  return 0;
}

void initializeHostArrays(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_uH[i][j]=th_complex(0.0f, 0.0f); h_u[i][j]=th_complex(0.0f, 0.0f);
    }
    h_c[i]=-a/2.0f;
  }
  h_c[0] /= 1+a;
  for (int i = 1; i<xmax; i++) {
    h_c[i] = h_c[i]/((1+a) + a*h_c[i-1]/2.0f);
  }
  h_u[xmax/2][xmax/2] = 10000;
  h_uH[xmax/2][xmax/2] = th_complex(10000.0f, 0.0f);
}
void printInitialVariables(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]) {
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
void printResult(th_complex h_uH[][xmax], th_complex h_u[][xmax], th_complex h_c[]) {
  ofstream fout("data.dat");
  ofstream fout2("real.dat");
  ofstream fout3("real_dev.dat");
  float konst = 1/296.4f;
  float citatel, menovatel,value,sum = 0;
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      citatel = pow((float)(i-xmax/2)/xmax,2) + pow((float)(j-xmax/2)/xmax,2);
      menovatel = 2.0f * D * tSec;
      value = (konst / tSec) * exp(-citatel / menovatel);
      sum+=value;
      fout << i << " " << j << " " <<  h_u[i][j].real() << endl;
      fout2 << i <<" "<< j <<" "<< value << endl;
    }
  }
  cout << "sum of prediction = " << sum << endl;
  fout.close(); fout2.close();
  for(int t = 0; t<tmax; t+=100) {
    fout3 << (float)t/tmax*tSec << " " << 2*2*D*(float)t/tmax*tSec << endl;
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
void stdDev_r(ofstream& r, float t, th_complex u[][xmax]) {
  th_complex sum = 0;
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j < xmax; j++) {
      sum += (float)(pow((float)(i-xmax/2)/xmax*l,2) + pow((float)(j-xmax/2)/xmax*l,2))*u[i][j]/10000.0f;
    }
  }
  r << tSec*t/tmax << " " << sum.real() << endl;
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
void altCPU(th_complex h_u[][xmax],
            th_complex h_uH[][xmax],
            th_complex h_V[][xmax],
            th_complex h_c[],
            float xmax,
            th_complex a,
            th_complex b) {
  for(int i=0; i<xmax;i++) {
    h_uH[i][0]  = (1.0f-a)*h_u[i][0] + a/2.0f*(h_u[i-1][0]+h_u[i+1][0]);
    //h_uH[i][0] += h_V[i][0];  //d_V je uz vynasobene konstantami v init funkcii
    h_uH[i][0] /= b;
    th_complex di;
    for(int j=1; j < xmax-1; j++) {
      //di  = h_V[i][j];
      di += (1.0f-a)*h_u[i][j] + a/2.0f*(h_u[i-1][j]+h_u[i+1][j]);
      h_uH[i][j] = (di+a/2.0f*h_uH[i][j-1])/(b+a/2.0f*h_c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      h_u[i][j]=h_uH[i][j]-h_c[j]*h_u[i][j+1];
    }
  }
}
