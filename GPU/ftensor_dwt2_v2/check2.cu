#include"check.h"
#include<cuda_runtime.h>
 void checkruntime(cudaError_t err){
	if(err!=cudaSuccess){
		printf("cuda runtime API error %d:%sd",(int)err,
				cudaGetErrorString(err));
		exit(-1);
	}
}
void checkkernel(){
	cudaError_t err=cudaGetLastError();
	if(err!=cudaSuccess){
		printf("cuda kernel launch error %d:%s\n",(int)err,cudaGetErrorString(err));
		exit(-1);
	}
}
void checkfft(cufftResult err){
	if(err!=CUFFT_SUCCESS){
		printf("cuda fft API error:%d\n",(int)err);
		exit(-1);

	}
}
void checkcusolverDn(cusolverStatus_t err){
	if(err!=CUSOLVER_STATUS_SUCCESS){
		printf("cuda cusolverDn API error:%d\n",(int)err);
		exit(-1);
	}
}
void checkcusparse(cusparseStatus_t err){
	if(err!=CUSPARSE_STATUS_SUCCESS){
		printf("cuda cusparse API error:%d\n",(int)err);
		exit(-1);
	}
}
