/**==============================dwt_batch====================================
 * create on: 2,Jun,2018
 * @author: haili
 *============================================================================
 */
#include "dwt.h"
__device__ void exch(int tid,int l,float* A,float* B){
	if(tid<l){
					B[tid+l]=A[tid];
					B[tid]=A[l-tid-1];
					B[tid+2*l]=A[l-tid-1];
					__syncthreads();
				}
}
__global__ void dwt_batch(float* A,const int len,const int batch,float* B){
	const int tidx=threadIdx.x;
	const int tid_n=blockDim.x;
	int tid=tidx;
	const int filterlen=8;
	const int declen=(len+filterlen-1)/2;
	__shared__ float lowfilter[filterlen];
	//lowfilter[]={0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106};
	lowfilter[7]=0.2303778133088964;lowfilter[6]=0.7148465705529154;
	lowfilter[5]=0.6308807679398597;lowfilter[4]=-0.0279837694168599;
	lowfilter[3]=-0.1870348117190931;lowfilter[2]=0.0308413818355607;
	lowfilter[1]=0.0328830116668852;lowfilter[0]=-0.0105974017850890;
	__shared__ float highfilter[filterlen];
   // highfilter[]={-0.0106,-0.0329,0.0308,0.1870,-0.0280,-0.6309,0.7148,-0.2304};
	highfilter[7]=-0.0105974017850890;highfilter[6]=-0.0328830116668852;
	highfilter[5]=0.0308413818355607;highfilter[4]=0.1870348117190931;
	highfilter[3]=-0.0279837694168599;highfilter[2]=-0.6308807679398597;
	highfilter[1]=0.7148465705529154;highfilter[0]=-0.2303778133088964;
	__shared__ float temp[THREADS];
	__shared__ float temp1[THREADS];
//	int i=0;
	switch(len)
	{
	case 1:{

		for(int i=0;i<4;i++){
		temp[tidx]=A[blockIdx.x];
	    float tmp=0;
		float tmp1=0;
		tid=tidx;
		while(tid<8){
		tmp+=temp[tid+i]*lowfilter[7-tid];
		tmp1+=temp[tid+i]*highfilter[7-tid];
		tid+=tid_n;
		}
                 __syncthreads();
		temp[tidx]=tmp;
		temp1[tidx]=tmp1;
		__syncthreads();
		int k=THREADS/2;
		while(k!=0){
			if(tidx<k){
				temp[tidx]+=temp[tidx+k];
				temp1[tidx]+=temp1[tidx+k];
			}
			__syncthreads();
			k/=2;
		  }
		if(tidx==0){
			B[i+blockIdx.x*8]=temp[tidx];
			B[i+4+blockIdx.x*8]=temp1[tidx];
		}
		__syncthreads();
		}
		break;
	   }
	case 2 :{
		for(int i=0;i<4;i++){
		float tmp=0;
		float tmp1=0;
		tid=tidx;
		if(tid<len){
			temp[tid+len]=A[tid+blockIdx.x*len];
			temp[tid]=A[len-tid-1+blockIdx.x*len];
			temp[tid+2*len]=A[len-tid-1+blockIdx.x*len];
				}
		__syncthreads();
		if(tid<len*3){
			temp1[tid+len*3]=temp[tid];
			temp1[tid]=temp[3*len-tid-1];
			temp1[tid+6*len]=temp[3*len-tid-1];
		}
		__syncthreads();
		while(tid<8){
		tmp+=temp1[tid+2+i*2]*lowfilter[7-tid];
		tmp1+=temp1[tid+2+i*2]*highfilter[7-tid];
		tid+=tid_n;
		}

		temp[tidx]=tmp;
		temp1[tidx]=tmp1;
		__syncthreads();
		int k=THREADS/2;
		while(k!=0){
		if(tidx<k){
		  temp[tidx]+=temp[tidx+k];
		  temp1[tidx]+=temp1[tidx+k];
					}
		 __syncthreads();
			k/=2;
			  }
		/*tmp=temp[0];
		printf("dshd%f\n",tmp);*/
		if(tidx==0){
		  B[i+blockIdx.x*8]=temp[tidx];
		  B[i+4+blockIdx.x*8]=temp1[tidx];
		 }
		__syncthreads();
		}
		break;
	}
	case 3 :{
		for(int i=0;i<5;i++){
			float tmp=0;
			float tmp1=0;
			tid=tidx;
			if(tid<len){
				temp[tid+len]=A[tid+blockIdx.x*len];
				temp[tid]=A[len-tid-1+blockIdx.x*len];
				temp[tid+2*len]=A[len-tid-1+blockIdx.x*len];
						}
			__syncthreads();
			if(tid<len*3){
				temp1[tid+len*3]=temp[tid];
				temp1[tid]=temp[3*len-tid-1];
				temp1[tid+6*len]=temp[3*len-tid-1];
			}
			__syncthreads();
			while(tid<8){
			tmp+=temp1[tid+6+i*2]*lowfilter[7-tid];
			tmp1+=temp1[tid+6+i*2]*highfilter[7-tid];
			tid+=tid_n;
			}
			temp[tidx]=tmp;
			temp1[tidx]=tmp1;
			__syncthreads();
			int k=THREADS/2;
			while(k!=0){
			if(tidx<k){
			temp[tidx]+=temp[tidx+k];
			temp1[tidx]+=temp1[tidx+k];
							}
			 __syncthreads();
			k/=2;
			}
			if(tidx==0){
   		  B[i+blockIdx.x*10]=temp[tidx];
   		  B[i+5+blockIdx.x*10]=temp1[tidx];
				 }
		__syncthreads();
				}
		break;
	}
	case 4 :{
		for(int i=0;i<5;i++){
			float tmp=0;
			float tmp1=0;
			tid=tidx;
			if(tid<len){
			temp[tid+len]=A[tid+blockIdx.x*len];
			temp[tid]=A[len-tid-1+blockIdx.x*len];
		    temp[tid+2*len]=A[len-tid-1+blockIdx.x*len];
						}
			__syncthreads();
			if(tid<len*3){
			temp1[tid+len*3]=temp[tid];
			temp1[tid]=temp[3*len-tid-1];
			temp1[tid+6*len]=temp[3*len-tid-1];
			}
			__syncthreads();
			while(tid<8){
			tmp+=temp1[tid+10+i*2]*lowfilter[7-tid];
			tmp1+=temp1[tid+10+i*2]*highfilter[7-tid];
			tid+=tid_n;
			}
			 temp[tidx]=tmp;
			 temp1[tidx]=tmp1;
			__syncthreads();
			int k=THREADS/2;
			while(k!=0){
				if(tidx<k){
				temp[tidx]+=temp[tidx+k];
				temp1[tidx]+=temp1[tidx+k];
							}
				__syncthreads();
				k/=2;
				}
			__syncthreads();
			tmp1=temp1[0];
	      if(tidx==0){
		   	B[i+blockIdx.x*10]=temp[tidx];
	    	 B[i+5+blockIdx.x*10]=temp1[tidx];
						 }
		  __syncthreads();
				}
		break;
	}
	case 5 :{
		for(int i=0;i<6;i++){
			float tmp=0;
			float tmp1=0;
			tid=tidx;
			temp[tidx]=0;
			temp1[tidx]=0;
			if(tid<len){
			temp[tid+len]=A[tid+blockIdx.x*len];
			temp[tid]=A[len-tid-1+blockIdx.x*len];
		    temp[tid+2*len]=A[len-tid-1+blockIdx.x*len];
							}
			__syncthreads();
		 if(tid<len*3){
			temp1[tid+len*3]=temp[tid];
			temp1[tid]=temp[3*len-tid-1];
			temp1[tid+6*len]=temp[3*len-tid-1];
					}
			__syncthreads();
		while(tid<8){
			tmp+=temp1[tid+14+i*2]*lowfilter[7-tid];
			tmp1+=temp1[tid+14+i*2]*highfilter[7-tid];
			tid+=tid_n;
					}
		    temp[tidx]=0;
		    temp1[tidx]=0;
		    __syncthreads();
			temp[tidx]=tmp;
			temp1[tidx]=tmp1;
			__syncthreads();
	    int k=THREADS/2;
	    while(k!=0){
			if(tidx<k){
			   temp[tidx]+=temp[tidx+k];
			   temp1[tidx]+=temp1[tidx+k];
									}
			 __syncthreads();
					k/=2;
						}
		 if(tidx==0){
			 	B[i+blockIdx.x*12]=temp[tidx];
			    B[i+6+blockIdx.x*12]=temp1[tidx];
								 }
		 __syncthreads();
						}
		break;
	}
	default:{
	for(int i=0;i<declen;i++){
		float tmp=0;
		float tmp1=0;
		tid=tidx;
		while(tid<filterlen){
		int p=2*i-tid+1;
		if((p<0) && (p>=(-filterlen+1))){
			tmp+=A[-p-1+blockIdx.x*len]*lowfilter[tid];
			tmp1+=A[-p-1+blockIdx.x*len]*highfilter[tid];
		}else {
			if((p>len-1) && (p<=len+filterlen-2)){
			tmp+=A[2*len-1-p+blockIdx.x*len]*lowfilter[tid];
			tmp1+=A[2*len-1-p+blockIdx.x*len]*highfilter[tid];
		}else{
			if((p>=0) && (p<=len-1)){
			tmp+=A[p+blockIdx.x*len]*lowfilter[tid];
			tmp1+=A[p+blockIdx.x*len]*highfilter[tid];
		}else{
			tmp+=0;
		    tmp1+=0;
		}
	         }
		           }
		//printf("%f\n",tmp);
		tid+=tid_n;
		}
		temp[tidx]=tmp;
		temp1[tidx]=tmp1;
		__syncthreads();
		int k=THREADS/2;
		while(k!=0){
			if(tidx<k){
				temp[tidx]+=temp[tidx+k];
				temp1[tidx]+=temp1[tidx+k];	
			}
                   __syncthreads();
			k/=2;
		}
		
	//	tmp=temp[0];
	 //   printf("%f\n",tmp);
		if(tidx==0){
		B[i+blockIdx.x*declen*2]=temp[tidx];
		B[i+declen+blockIdx.x*declen*2]=temp1[tidx];
		}
           __syncthreads();
	}
	break;
	}
	}
}
/**
 * ================================idwt1_batch===============================
 * @A.length=len * batch
 * @B.length=((len/2==0)?2*len-filterlen+2:2*len-filterlen+1)*batch
 * @len=declen_L or  declen_H
 * ==========================================================================
 */
__global__ void idwt_batch(float* A,const int len,const int reclen,const int batch,float* B){
	const int tidx=threadIdx.x;
	const int tid_n=blockDim.x;
	int tid=tidx;
	const int filterlen=8;
	__shared__ float lowfilter[filterlen];
	//lowfilter[]={0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106};
	lowfilter[0]=0.2303778133088964;lowfilter[1]=0.7148465705529154;
	lowfilter[2]=0.6308807679398597;lowfilter[3]=-0.0279837694168599;
	lowfilter[4]=-0.1870348117190931;lowfilter[5]=0.0308413818355607;
	lowfilter[6]=0.0328830116668852;lowfilter[7]=-0.0105974017850890;

	__shared__ float highfilter[filterlen];
	 // highfilter[]={-0.0106,-0.0329,0.0308,0.1870,-0.0280,-0.6309,0.7148,-0.2304};
	highfilter[0]=-0.0105974017850890;highfilter[1]=-0.0328830116668852;
	highfilter[2]=0.0308413818355607;highfilter[3]=0.1870348117190931;
	highfilter[4]=-0.0279837694168599;highfilter[5]=-0.6308807679398597;
	highfilter[6]=0.7148465705529154;highfilter[7]=-0.2303778133088964;
	__shared__ float temp[THREADS];
//	const int reclen=2*len-filterlen+2;
	for(int i=0;i<reclen;i++){
		tid=tidx;
		float tmp=0;
		while(tid<len){
			int p=i-2*tid+filterlen-2;
			if((p>=0) && (p<filterlen)){
			 tmp+=lowfilter[p]*A[tid+blockIdx.x*2*len]+highfilter[p]*A[tid+len+blockIdx.x*2*len];
			}
			tid+=tid_n;
		}
		temp[tidx]=tmp;
		__syncthreads();
		int k=THREADS/2;
		while(k!=0){
			if(tidx<k){
				temp[tidx]+=temp[tidx+k];	
			}
                  __syncthreads();
			k/=2;
		}
		if(tidx==0){
			B[i+blockIdx.x*reclen]=temp[tidx];
		}
		__syncthreads();
	}
}
