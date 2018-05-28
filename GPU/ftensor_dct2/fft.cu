#include"fft.h"
void fft(int m,int n,int k,int l,cufftComplex* t,cufftComplex* ft){
	int a=m*n*l;//CUFFT_INVERSE OR CUFFT_FORWARD
	cufftComplex* odata;
	cufftComplex* idata;
	cudaError_t stat1=cudaSuccess;
	cudaError_t stat2=cudaSuccess;
	cudaError_t stat3=cudaSuccess;
	cudaError_t stat4=cudaSuccess;
	cufftResult cufftstat1=CUFFT_SUCCESS;
	cufftResult cufftstat2=CUFFT_SUCCESS;
	cufftResult cufftstat3=CUFFT_SUCCESS;
	stat1=cudaMalloc((void**)&odata,sizeof(cufftComplex)*m*n*k*l);
	stat2=cudaMalloc((void**)&idata,sizeof(cufftComplex)*m*n*k*l);
	assert(stat1==cudaSuccess);
	assert(stat2==cudaSuccess);
	stat3=cudaMemcpy(
			idata,
			t,
			sizeof(cufftComplex)*m*n*k*l
			,cudaMemcpyHostToDevice);
	assert(cudaSuccess==stat3);
	cufftHandle plan;
	cufftstat1=cufftPlan2d(&plan,a,k,CUFFT_C2C);
    cufftstat2=cufftExecC2C(
    		plan,
    		(cufftComplex*)idata,
    		(cufftComplex*)odata,
    		CUFFT_FORWARD);
    cudaDeviceSynchronize();
    stat4=cudaMemcpy(
    		ft,
    		odata,
    		sizeof(cufftComplex)*m*n*k*l,
    		cudaMemcpyDeviceToHost);
    assert(stat4==cudaSuccess);
    cufftstat3=cufftDestroy(plan);
    if(cufftstat1!=CUFFT_SUCCESS||
    		cufftstat2!=CUFFT_SUCCESS||
    		cufftstat3!=CUFFT_SUCCESS){
    	printf("cufft API error");
    	exit(-1);
    }
    cudaFree(odata);
    cudaFree(idata);
}
void ifft(int m ,int n,int k,int l,cufftComplex* t,cufftComplex* ft){
	int a=m*n*l;//CUFFT_INVERSE OR CUFFT_FORWARD
		cufftComplex* odata;
		cufftComplex* idata;
		cudaError_t stat1=cudaSuccess;
		cudaError_t stat2=cudaSuccess;
		cudaError_t stat3=cudaSuccess;
		cudaError_t stat4=cudaSuccess;
		cufftResult cufftstat1=CUFFT_SUCCESS;
		cufftResult cufftstat2=CUFFT_SUCCESS;
		cufftResult cufftstat3=CUFFT_SUCCESS;
		stat1=cudaMalloc((void**)&odata,sizeof(cufftComplex)*m*n*k*l);
		stat2=cudaMalloc((void**)&idata,sizeof(cufftComplex)*m*n*k*l);
		stat3=cudaMemcpy(
				idata,
				t,
				sizeof(cufftComplex)*m*n*k*l,
				cudaMemcpyHostToDevice);
		cufftHandle plan;
		cufftstat1=cufftPlan2d(&plan,a,k,CUFFT_C2C);
	    cufftstat2=cufftExecC2C(
	    		plan,
	    		(cufftComplex*)idata,
	    		(cufftComplex*)odata,
	    		CUFFT_INVERSE);
	    if(cudaDeviceSynchronize()!=cudaSuccess){
	    	printf("cuda synchronize failed");
	    	return;
	    }
	    stat4=cudaMemcpy(
	    		ft,
	    		odata,
	    		sizeof(cufftComplex)*m*n*k*l,
	    		cudaMemcpyDeviceToHost);
	    cufftstat3=cufftDestroy(plan);
	    assert(stat1==cudaSuccess);
	    assert(stat2==cudaSuccess);
	    assert(stat3==cudaSuccess);
	    assert(stat4==cudaSuccess);
	    assert(cufftstat1==CUFFT_SUCCESS);
	    assert(cufftstat2==CUFFT_SUCCESS);
	    assert(cufftstat3==CUFFT_SUCCESS);
	    cudaFree(odata);
	    cudaFree(idata);
}
