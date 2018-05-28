#include"fft.h"
void fft_batch(int m,int n,int k,int l,cuComplex* t,cuComplex* ft)
{
	cufftHandle plan;
	int rank=2;
	/*int N[rank-1]={l,k};
	int inembed[rank-1]={k,l};
	int onembed[rank-1]={k,l};*/
	int N[rank-1]={0};
	N[0]=l;N[1]=k;
		int inembed[rank-1]={0};
		inembed[0]=l;
		inembed[1]=k;
		int onembed[rank-1]={0};
		onembed[0]=l;
		onembed[1]=k;
	int istride=1;
	int ostride=1;
	int idist=k*l;
	int odist=k*l;
	cufftType type=CUFFT_C2C;
	int batch=m*n;
	cufftResult_t cufftstat1=CUFFT_SUCCESS;
	cufftResult_t cufftstat2=CUFFT_SUCCESS;
	cufftResult_t cufftstat3=CUFFT_SUCCESS;
	cudaError_t stat1=cudaSuccess;
	cudaError_t stat2=cudaSuccess;
	cudaError_t stat3=cudaSuccess;
	cudaError_t stat4=cudaSuccess;
	cudaError_t stat5=cudaSuccess;
	cufftstat1=cufftPlanMany(
			&plan,
			rank,
			N,
			inembed,
			istride,
			idist,
			onembed,
			ostride,
			odist,
			type,
			batch
			);
    stat1=cudaDeviceSynchronize();
    cufftComplex* idata;cufftComplex* odata;
    stat2=cudaMalloc((void**)&idata,sizeof(cufftComplex)*m*n*k*l);
    stat3=cudaMalloc((void**)&odata,sizeof(cufftComplex)*m*n*k*l);
    stat4=cudaMemcpy(
    		idata,
    		t,
    		sizeof(cufftComplex)*m*n*k*l,
    		cudaMemcpyHostToDevice);
    int direction=CUFFT_FORWARD;
    cufftstat2=cufftExecC2C(
            plan,
    		idata,
    		odata,
    		direction);
    stat5=cudaMemcpy(
    		ft,
    		odata,
    		sizeof(cufftComplex)*n*m*k*l,
    		cudaMemcpyDeviceToHost);
    cufftstat3=cufftDestroy(plan);
    if(
    		stat1!=cudaSuccess||
    		stat2!=cudaSuccess||
    		stat3!=cudaSuccess||
    		stat4!=cudaSuccess||
    		stat5!=cudaSuccess){
    	printf("cuda runtime API error");
    	return;
    }
    if(
    		cufftstat1!=CUFFT_SUCCESS||
    		cufftstat2!=CUFFT_SUCCESS||
    		cufftstat3!=CUFFT_SUCCESS){
    	printf("cufft API error");
    	return;
    }

    stat1=cudaFree(idata);
    assert(stat1==cudaSuccess);
    stat1=cudaFree(odata);
    assert(stat1==cudaSuccess);

}
void ifft_batch(int m,int n,int k,int l,cuComplex* t,cuComplex* ft)
{
	cufftHandle plan;int rank=2;
/*nt N[rank-1]={l,k};
	int inembed[rank-1]={k,l};int onembed[rank-1]={k,l};*/
	int istride=1;int ostride=1;
	int N[rank-1]={0};
		N[0]=l;N[1]=k;
			int inembed[rank-1]={0};
			inembed[0]=l;
			inembed[1]=k;
			int onembed[rank-1]={0};
			onembed[0]=l;
			onembed[1]=k;
	int idist=k*l; int odist=k*l;
	cufftType type=CUFFT_C2C;
	int batch=m*n;
	    cufftResult_t cufftstat1=CUFFT_SUCCESS;
		cufftResult_t cufftstat2=CUFFT_SUCCESS;
		cufftResult_t cufftstat3=CUFFT_SUCCESS;
		cudaError_t stat1=cudaSuccess;
		cudaError_t stat2=cudaSuccess;
		cudaError_t stat3=cudaSuccess;
		cudaError_t stat4=cudaSuccess;
		cudaError_t stat5=cudaSuccess;
	cufftstat1=cufftPlanMany(
			&plan,
			rank,
			N,
			inembed,
			istride,
			idist,
			onembed,
			ostride,
			odist,
			type,
			batch
			);
    stat1=cudaDeviceSynchronize();
    cufftComplex* idata;cufftComplex* odata;
    stat2=cudaMalloc((void**)&idata,sizeof(cufftComplex)*m*n*k*l);
    stat3=cudaMalloc((void**)&odata,sizeof(cufftComplex)*m*n*k*l);
    stat4=cudaMemcpy(
    		idata,
    		t,
    		sizeof(cufftComplex)*m*n*k*l
    		,cudaMemcpyHostToDevice);
    int direction=CUFFT_INVERSE;
    cufftstat2=cufftExecC2C(
            plan,
    		idata,
    		odata,
    		direction);
    stat5=cudaMemcpy(
    		ft,
    		odata,
    		sizeof(cufftComplex)*n*m*k*l,
    		cudaMemcpyDeviceToHost);
    cufftstat3=cufftDestroy(plan);
    assert(stat1==cudaSuccess);
    assert(stat2==cudaSuccess);
    assert(stat3==cudaSuccess);
    assert(stat4==cudaSuccess);
    assert(stat5==cudaSuccess);
    if(
    		cufftstat1!=CUFFT_SUCCESS||
    		cufftstat2!=CUFFT_SUCCESS||
    		cufftstat3!=CUFFT_SUCCESS){
    	printf("cufft API error");
    	exit(-1);

    }
    stat1=cudaFree(idata);
    assert(stat1==cudaSuccess);
    stat1=cudaFree(odata);
    assert(stat1==cudaSuccess);

}
