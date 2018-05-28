/*=================dct2=====================================*/
/**
 * Create:22 May,2018
 *
 * @author:haili
 *
 */
#include "dct.cuh"
__global__ void dct2_batch2(float* A,const int m,const int n,float* B){
	__shared__ float mat[THREADS];
	const int tidx=threadIdx.x;
	int tid=tidx;
	int bidx=blockIdx.x;
	const int t_n=blockDim.x;
	double a=0;
	double b=0;
    double temp=0;
  //  int i=0;
   // int j=0;
    while(bidx<BLOCKS){
    for(int i=0;i<m;i++){
    	for(int j=0;j<n;j++){
    tid=tidx;
    temp=0;
    while(tid<m*n){
           int row=tid/n;
           int col=tid%n;
            temp+=A[row*n+col+blockIdx.x*m*n]*cos(pi*(row+0.5)*i/m)*cos(pi*(col+0.5)*j/n);
            tid+=t_n;
    }
    mat[tidx]=temp;
    __syncthreads();
    int k=THREADS/2;
       while(k!=0){
       	if(tidx<k){
       	     mat[tidx]+=mat[tidx+k];
       	     __syncthreads();
       		}
       	    k/=2;
       	}
       a=0;
       b=0;
       if(i==0){
         	a=(double)1/sqrt(double(m));
       }else{
           	a=sqrt((double)2)/sqrt(double(m));
          }
       if(j==0){
               b=(double)1/sqrt(double(n));
       }else{
               b=sqrt((double)2)/sqrt(double(n));
           }
       if(tidx==0){
           	B[i*n+j+blockIdx.x*m*n]=a*b*mat[tidx];
           }
           __syncthreads();

       }
   }
    bidx+=gridDim.x;
    }
}
/*======================idct================================*/
/**
 * @create:22 May,2018
 * @author: haili
 * idct2 shared memmry of size is :THREADS
 * */
__global__ void idct2_batch2(float* A,const int m,const int n,float* B){
	__shared__ float mat[THREADS];
	const int tidx=threadIdx.x;
	int bidx=blockIdx.x;
	const int t_n=blockDim.x;
	const int b_n=gridDim.x;
//	int i=0;
//   int j=0;
	while(bidx<BLOCKS){
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
	int tid=tidx;
	double temp=0;
	double a=0;
	double b=0;
	while(tid<m*n){
		int row=tid/n;
		int col=tid%n;
		if(row==0){
			a=(double)1/sqrt((double)m);
		}else{
			a=sqrt((double)2)/sqrt((double)m);
		}
		if(col==0){
			b=(double)1/sqrt((double)n);
		}else{
			b=sqrt((double)2)/sqrt((double)n);
		}
		temp+=a*b*A[row*n+col+blockIdx.x*m*n]*cos(pi*(i+0.5)*row/m)*cos(pi*(j+0.5)*col/n);
		tid+=t_n;
	}
	mat[tidx]=temp;
	__syncthreads();
	int k=THREADS/2;
	while(k!=0){
		if(tidx<k){
			mat[tidx]+=mat[tidx+k];
			__syncthreads();
		}
		k/=2;
	}

	if(tidx==0){
		B[i*n+j+blockIdx.x*m*n]=mat[tidx];
	}
	__syncthreads();
		}
	}
	bidx+=b_n;
	}
}
