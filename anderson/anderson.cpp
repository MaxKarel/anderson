#include <iostream>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <cstdlib>
#include <thrust/complex.h>
using namespace std;

typedef thrust::complex<double> thComp;

const int xmax = 201;
const int tmax = 500;
float dt = 0.02;
//float dx = 2*sqrt(dt);
float scale = 2;
float dx = 1;
thComp imag_one (0.0f, 1.0f);
thComp a = -imag_one*thComp(dt/(dx*dx),0.0);
thComp b = 1.0+a;

void printInitialVariables(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax], thComp h_c[]);
void printResult(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax], thComp h_c[]);
void initializeHostArrays(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax], thComp h_c[]);
void stdDev_r(ofstream& r, float t, thComp u[][xmax]);
void transpose(thComp arr[][xmax]);
void altCPU(thComp h_u[][xmax], thComp h_V[][xmax], thComp h_c[],
            int xmax, thComp a, thComp b);

  thComp h_u[xmax][xmax] = {}; //alocating on heap
  thComp h_uH[xmax][xmax] = {};
  thComp h_V[xmax][xmax] = {};
  thComp h_c[xmax] = {};

int main() {

  if(xmax > 1024) {printf("Size of arr is greater than maximal number of threads %i\n", xmax);}

  initializeHostArrays(h_u, h_uH, h_V, h_c);

  ofstream r_fout("std.cpp");
  //////////////////////////////////////////////
  ////  CPU                                 ////
  //////////////////////////////////////////////
  for (int t = 0; t< tmax; t++) {
    altCPU(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    altCPU(h_u, h_V, h_c, xmax, a, b);
    transpose(h_u);
    transpose(h_V);
    if (t%10==0) {
      stdDev_r(r_fout,t,h_u);
    }
  }

  printResult(h_u, h_uH, h_V, h_c);
  return 0;
}

void initializeHostArrays(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax], thComp h_c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      h_u[i][j] =thComp(0.0, 0.0);
      h_uH[i][j]=thComp(0.0, 0.0);
      h_V[i][j] =thComp(2*scale*(float)(rand()%10000/10000.0-0.5), 0.0);
      h_V[i][j] *= thComp(dt,0.0)/imag_one;
      //h_V[i][j] = 0;
    }
  }
  //Nastavenie pociatocnych podmienok
  h_u[xmax/2][xmax/2] = thComp(1.0, 0);
}
void printInitialVariables(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax], thComp h_c[]) {
  cout << "     dx== " << dx <<endl<< "     dt== " << dt << endl << "      a== " << a << endl;
  cout << "      b== " << b << endl;
}
void printResult(thComp h_u[][xmax], thComp h_uH[][xmax], thComp h_V[][xmax],thComp h_c[]) {
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
void stdDev_r(ofstream& r, float t, thComp u[][xmax]) {
  thComp sum = 0;
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j < xmax; j++) {
      sum += (float)(pow((float)(i-xmax/2),2) + pow((float)(j-xmax/2),2))*(u[i][j].real()*u[i][j].real() + u[i][j].imag()*u[i][j].imag());
    }
  }
  r << t/tmax << " " << sum.real() << endl;
}
void transpose(thComp arr[][xmax]) {
  thComp help;
  for(int i = 0; i < xmax; i++) {
    for(int j = i+1; j < xmax; j++) {
      help = arr[i][j];
      arr[i][j] = arr[j][i];
      arr[j][i] = help;
    }
  }
}

void altCPU(thComp h_u[][xmax], thComp h_V[][xmax], thComp h_c[],
            int xmax, thComp a, thComp b) {

  thComp mod_rs[xmax];  //modified right side

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
    thComp di;  //unmodified right side, help variable
    for(int j=1; j < xmax-1; j++) {
      di  = (1.0-a)*h_uH[i][j] + a/2.0*(h_uH[i-1][j]+h_uH[i+1][j]);
      mod_rs[j] = (di+a/2.0*mod_rs[j-1])/((b - h_V[i][j])+a/2.0*h_c[j-1]);
    }
    for(int j=xmax-2; j>0; j--) {
      h_u[i][j]=mod_rs[j]-h_c[j]*h_u[i][j+1];
    }
  }
  cout.precision(17);
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

