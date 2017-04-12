#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

const int N = 1024;
const int blocksize = sqrt(N);
const int dim = 2;
__global__
void hello(int *a, int* i_d)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	a[id] = i_d[0];
	//a[id] = 1;
	i_d[0]++;
}

int main()
{
	int* a;
	int* b;
	int* c;
	int* ad;
	int* bd;
	int* cd;
	int* i_d;
	int isize = N*sizeof(int);

	a = (int*)malloc(isize);
	b = (int*)malloc(isize);
	c = (int*)malloc(isize);
	for(int i=0; i<N;i++) {
		a[i]=0;
		b[i]=0;
		c[i]=0;
	}
	cudaMalloc( &ad, isize );
	cudaMalloc( &bd, isize );
	cudaMalloc( &cd, isize );
	cudaMalloc( &i_d, sizeof(int) );
	int* i_h = new int;
	*i_h = 0;
	cudaMemcpy( ad, a, isize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );
	cudaMemcpy( cd, c, isize, cudaMemcpyHostToDevice );
	cudaMemcpy( i_d, i_h, sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimBlock( N/1024 );
	dim3 dimGrid( 1024 );
	hello<<<dimGrid, dimBlock>>>(ad, i_d);
	cudaMemcpy( a, ad, isize, cudaMemcpyDeviceToHost );
	cudaMemcpy( b, bd, isize, cudaMemcpyDeviceToHost );
	cudaMemcpy( c, cd, isize, cudaMemcpyDeviceToHost );
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	cudaFree(i_d);
	int formatOpt = 2;
	formatOpt *= 16;
	for (int i=0; i<=N/formatOpt;i++) {
    for (int j=0;j<formatOpt;j++){
      if ( formatOpt*i+j<N ) {cout << a[formatOpt*i+j] << " ";}
    }
    cout << endl;
  }

	delete[] a;
	return 0;
}
