#include"kernel.h"
/**
 * @odd_even sort
 *  Created on: Apr 16, 2018
 *  len is even
 * @author: haili
 */
__device__ void swap(cdata *a,cdata *b){
   cdata c;
   c=*a;
   *a=*b;
   *b=c;
}

__global__ void odd_even(float* s,cdata* A,int len,int r){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	cdata a;
        
         for(int i=0;i<len/2;i++){
         if(2*tid+1<len){
         if(A[2*tid].a>A[2*tid+1].a){
            a=A[2*tid]; 
            A[2*tid]=A[2*tid+1];
            A[2*tid+1]=a;
          }
          }
         __syncthreads();
         if(2*tid+2<len){
           if(A[2*tid+1].a>A[2*tid+2].a){
            a=A[2*tid+1]; 
            A[2*tid+1]=A[2*tid+2];
            A[2*tid+2]=a;
           }
          }
          __syncthreads();
         }
         for(int i=0;i<r;i++){
          s[A[i].b]=0;
         }
	}
/**
 * @device tensor_scalar
 * Create on:Apr 17 2018
 * @author: haili
 * the size of tensor is m×n×k×l
 */
__global__ void d_tensor_scalar(cuComplex* A,cuComplex* T,
		const int m,const int n,const int k,const int l){

	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*k*l){
		A[(tid%(m*n))*k*l+tid/(m*n)]=T[tid];
		tid+=t_n;
		__syncthreads();
	}
}
/**
 * @device tensor_scalar to tensor
 * Create on:Apr 17 2018
 * @author: haili
 * the size of tensor is m×n×k×l
 */

__global__ void d_tensor_scalar_tensor(cuComplex* A,cuComplex* T,
		const int m,const int n,const int k,const int l){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*k*l){
		A[tid]=T[(tid%(m*n))*k*l+tid/(m*n)];
		tid=tid+t_n;
		__syncthreads();
	}

}
/**
 * @device matview_transopse
 * Create on:Apr 17 2018
 * @author: haili
 * the size of tensor is m×n×k×l
 */
__global__ void d_matview_transpose(cuComplex* A,cuComplex* T,const int m,
		const int n,const int k,const int l){
	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<m*n*k*l){
		A[(tid/(m*n))*n*m+(tid%(m*n))/n+((tid%(m*n))%n)*m].x=T[tid].x;
		__syncthreads();
		A[(tid/(m*n))*n*m+(tid%(m*n))/n+(tid%n)*m].y=0-T[tid].y;
		tid=tid+t_n;
		__syncthreads();
	}

}
/**
 * @device csr to matview
 * Create on:Apr 25 2018
 * @author: haili
 * the size of matview is m*k*l × n*k*l
 */
__global__ void d_csrtomatview(cuComplex* A,cuComplex* data,const int* row,
		const int* col, const int m,const int n,const int k,const int l){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	int num=0;

	while(tid<m*k*l){
		if(tid<m*k*l){
		num=row[tid+1]-row[tid];
		__syncthreads();
		for(int i=row[tid];i<row[tid]+num;i++){
			/*A[tid*n+col[i]].x=(float)data[i].x/(k*l);
			A[tid*n+col[i]].y=(float)data[i].y/(k*l);*/
			A[tid*n+col[i]%n]=data[i];
			__syncthreads();

		}
		}
		tid=tid+t_n;
}
}
__global__ void d_zero(cuComplex* A,int len){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int t_n=blockDim.x*gridDim.x;
	while(tid<len){
		A[tid].x=0;
		A[tid].y=0;
		tid+=t_n;
	}
	__syncthreads();

}
