/**
 * @device matview_transopse
 * Create on:Apr 17 2018
 * @author: haili
 * the size of tensor is m×n×k×l
 */
__global__ void d_batch_transpose(float* A,float* T,const int m,
		const int n,const int batch){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*batch){
		A[(tid/(m*n))*n*m+(tid%(m*n))/n+((tid%(m*n))%n)*m]=T[tid];
		tid=tid+t_n;
		__syncthreads();
	}
   

}
