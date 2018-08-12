#include "dct.h"
/**
 * implement  a set of dct with the same size.
 * create:19 May 2018
 * @author: haili
 *threadnum must be even ,threadnum < len
 */
 __global__ void dct_batch(float* A,const int len,const int batch,float* B){
         __shared__  float tmp[THREADS];
        const int tidx=threadIdx.x;
	const int t_n=blockDim.x;
        const int bidx=blockIdx.x;
        const int b_n=gridDim.x;
        int Bid=bidx;
	double temp=0.0;
	double a=0.0;
        while(Bid<batch){
	for(int i=0;i<len;i++){
		int Tid2=tidx;
		temp=0.0;
		a=0;
		if(i==0){
			a=(double)1/sqrt((double)len);
		}else{
			a=sqrt((double)2)/sqrt((double)len);
		}
		while(Tid2<len){
		//temp+=A[Tid2+blockIdx.x*len]*cos(pi*(Tid2+0.5)*i/len);
                temp+=A[Tid2+Bid*len]*cos(pi*(Tid2+0.5)*i/len);
		Tid2+=t_n;
		}
		tmp[tidx]=temp;
		__syncthreads();
	    int k=THREADS/2;
		while(k!=0){
			if(tidx<k){
				tmp[tidx]+=tmp[tidx+k];
			}
			__syncthreads();
			k/=2;
		  }
		if(tidx==0){
		//	B[i+blockIdx.x*len]=a*tmp[tidx];
                      B[i+Bid*len]=a*tmp[tidx];
			}
}            Bid+=b_n;
	}

}
 /*======================idct================================*/
 /**
  * @create:22 May,2018
  * @author: haili
  * idct2 shared memmry of size is :THREADS
  * */
 __global__ void idct_batch(float* A,const int len,const int batch,float* B){
 	__shared__ float mat[THREADS];
 	const int tidx=threadIdx.x;
 	int t_n=blockDim.x;
        const int bidx=blockIdx.x;
        const int b_n=gridDim.x;
        int Bid=bidx;
        while(Bid<batch){
 	for(int j=0;j<len;j++){
 	int tid=tidx;
 	double temp=0;
 	double a=0;
 	while(tid<len){
 		if(tid==0){
 			a=(double)1/sqrt((double)len);
 		}else{
 			a=sqrt((double)2)/sqrt((double)len);
 		}
 	//	temp+=a*A[tid+blockIdx.x*len]*cos((pi*(j+0.5)*tid)/len);
               temp+=a*A[tid+Bid*len]*cos((pi*(j+0.5)*tid)/len);
 		tid+=t_n;
 	}
 	mat[tidx]=temp;
 	__syncthreads();
 	int k=THREADS/2;
 	while(k!=0){
 		if(tidx<k){
 			mat[tidx]+=mat[tidx+k];
 		}
 		__syncthreads();
 		k/=2;
 	}

 	if(tidx==0){
 	//	B[j+blockIdx.x*len]=mat[tidx];
            B[j+Bid*len]=mat[tidx];
 	}
 		}
         Bid+=b_n;
        }
 	}


