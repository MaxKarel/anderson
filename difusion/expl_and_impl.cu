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
*/
