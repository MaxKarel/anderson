#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;


const int xmax = 401;
int tmax = 2000;
double tSec = 0.1;
double D = 0.5;
double l = 1;
double dx = l / (double)xmax;
double dt = tSec / (double)tmax;
double a = (D*dt)/(dx*dx);
double b = 1+2*a;
double c[xmax];

void impl(int i, bool row, double uH[][xmax], double u[][xmax]);
void initializeHostArrays(double uH[][xmax], double u[][xmax], double c[]);
void printInitialVariables(double uH[][xmax], double u[][xmax], double c[]);
void printResult(double uH[][xmax], double u[][xmax], double c[]);

int main() {
  double uH[xmax][xmax];
  double u[xmax][xmax];
  initializeHostArrays(u, uH, c);
  //printInitialVariables(uH, u, c);
  for (int t = 0; t< tmax; t++) {
    if(t%2 == 0) {
      for (int i = 0; i < xmax; i++) {
        copy(&u[0][0],&u[0][0]+xmax*xmax,&uH[0][0]);
        impl(i,false, uH, u);
      }
    } else {
      for (int i =0; i < xmax; i++) {
        copy(&u[0][0],&u[0][0]+xmax*xmax,&uH[0][0]);
        impl(i,true, uH, u);
      }
    }
  }
  printResult(u, uH, c);
  return 0;
}

void impl(int i, bool row, double uH[][xmax], double u[][xmax]) {

  if (row) {
    for(int j = 1; j<xmax-1; j++) {
      u[i][j] = a*uH[i][j-1]+(1.0-2.0*a)*uH[i][j]+a*uH[i][j+1];
    }
    u[i][0] = u[i][0]/b;
    for(int j=1; j < xmax-1; j++) {
      u[i][j] = (u[i][j]+a*u[i][j-1])/(b+a*c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      u[i][j]=u[i][j]-c[j]*u[i][j+1];
    }
  } else {
    for(int j = 1; j<xmax-1; j++) {
      u[j][i]= a*uH[j-1][i]+(1.0-2.0*a)*uH[j][i]+a*u[j+1][i];
    }
    u[0][i] = u[0][i]/b;
    for(int j=1; j < xmax-1; j++) {
      u[j][i] = (u[j][i]+a*u[j-1][i])/(b+a*c[j-1]);
    }
    for(int j=xmax-2; j>-1; j--) {
      u[j][i]=u[j][i]-c[j]*u[j+1][i];
    }
  }

}


void initializeHostArrays(double uH[][xmax], double u[][xmax], double c[]) {
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      uH[i][j]=0; u[i][j]=0;
    }
    c[i]=-a;
  }
  c[0] /= b;
  for (int i = 1; i<xmax; i++) {
    c[i] = c[i]/(double)(b + a*c[i-1]);
  }
  u[xmax/2][xmax/2] = 10000;
  uH[xmax/2][xmax/2] = 10000;
}
void printInitialVariables(double uH[][xmax], double u[][xmax], double c[]) {
  cout << " dx==" << dx <<endl<< " dt==" << dt << endl << " a==" << a << endl;
  cout << "b==" << b << endl;
  cout << "ARR C  " << endl << endl;
  for (int i = 0; i< xmax/16 + 1; i++) {
    for(int j= 0; j<16 ; j++) {
      if ((i*32 + j)<xmax){cout<<c[i*32 + j] << " ";}
    }
    cout << endl;
  }
  for(int i = 0; i< xmax; i++) {
    for(int j = 0; j< xmax; j++) {
      //cout<<u[i][j] << " ";
    }
    //cout <<endl;
  }
}
void printResult(double uH[][xmax], double u[][xmax], double c[]) {
  ofstream fout("data.dat");
  for(int i = 0; i<xmax; i++){
    for (int j = 0; j < xmax; j++) {
      fout << i << " " << j << " " <<  u[i][j] << " " << endl;
    }
  }
  fout.close();
  cout << u[50][50] << endl;
  double sum = 0;
  for(int i=0; i<xmax; i++){
    for(int j = 0; j<xmax; j++) {
      sum += u[i][j];
    }
  }
  cout << "sum == " << sum << endl;
  printInitialVariables(uH, u, c);
}
