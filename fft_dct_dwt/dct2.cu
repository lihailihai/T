/**
*Create:24 7 2018
*@author:haili
* dim block(x,y)
**/
//----------------------------------------------------------------
#include"dct.h"
__global__ void dct2(const float* A,const int m,const int n,float* B){
        __shared__ float mat[32][32+1];
        const int tid_x=threadIdx.x;
        const int tid_y=threadIdx.y;
        const int t_x=blockDim.x;
        const int t_y=blockDim.y;
        const int bid_x=blockIdx.x;
        float temp=0;
        int tidx=tid_x;
        int tidy=tid_y;	
	for(tidy=tid_y;tidy<n;tidy+=t_y){
        temp=0;
	for(tidx=tid_x;tidx<n*m;tidx+=t_x){
		int row=tidx/n;
                int col=tidx%n;
                temp+=A[row*n+col]*cos(pi*(row+0.5)*bid_x/m)*cos(pi*(col+0.5)*tidy/n); 
	}
	   mat[tid_y][tid_x]=temp;
           __syncthreads();
          int k=t_x/2;
  	  while(k!=0){
		if(tid_x<k){
		mat[tid_y][tid_x]+=mat[tid_y][tid_x+k];	
			}
            __syncthreads();
		k/=2;
	     }
	   float a=0;
		float b=0;
		if(bid_x==0){
			a=(float)1/sqrt(float(m));
			}else{
			a=sqrt((float)2)/sqrt(float(m));
			}
		if(tidy==0){
			b=(float)1/sqrt(float(n));
			}else{
			b=sqrt((float)2)/sqrt(float(n));
			}
	if(tid_x==0){
		B[bid_x*n+tidy]=a*b*mat[tid_y][tid_x];	
		}
	}
}


__global__ void idct2(const float* A,const int m,const int n,float* B){
	 __shared__ float mat[32][32+1];
        const int tid_x=threadIdx.x;
        const int tid_y=threadIdx.y;
        const int t_x=blockDim.x;
        const int t_y=blockDim.y;
        const int bid_x=blockIdx.x;
        float temp=0;
        int tidx=tid_x;
        int tidy=tid_y;
	for(tidy=tid_y;tidy<n;tidy+=t_y){
        temp=0;
	for(tidx=tid_x;tidx<n*m;tidx+=t_x){
		int row=tidx/n;
                int col=tidx%n;
		float a=0;
		float b=0;
		if(row==0){
			a=(float)1/sqrt(float(m));
			}else{
			a=sqrt((float)2)/sqrt(float(m));
			}
		if(col==0){
			b=(float)1/sqrt(float(n));
			}else{
			b=sqrt((float)2)/sqrt(float(n));
			}
                temp+=a*b*A[row*n+col]*cos(pi*(bid_x+0.5)*row/m)*cos(pi*(tidy+0.5)*col/n); 
	}
	   mat[tid_y][tid_x]=temp;
           __syncthreads();
          int k=t_x/2;
  	  while(k!=0){
		if(tid_x<k){
		mat[tid_y][tid_x]+=mat[tid_y][tid_x+k];	
			}
            __syncthreads();
		k/=2;
	     }
	  
	if(tid_x==0){
		B[bid_x*n+tidy]=mat[tid_y][tid_x];	
		}
	}
	
}
