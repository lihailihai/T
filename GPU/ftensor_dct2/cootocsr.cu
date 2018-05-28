#include "cootocsr.h"
void cootocsr(int m,int nnz,int* c,int* rrow){
	cusparseHandle_t handle;
	cudaError_t stat1=cudaSuccess;
	cudaError_t stat2=cudaSuccess;
	cudaError_t stat3=cudaSuccess;
	cudaError_t stat4=cudaSuccess;
	cusparseStatus_t stat6=CUSPARSE_STATUS_SUCCESS;
	cusparseStatus_t stat5=CUSPARSE_STATUS_SUCCESS;
	cusparseStatus_t stat7=CUSPARSE_STATUS_SUCCESS;
	int* row;
	int* crow;
	stat1=cudaMalloc((void**)&crow,sizeof(int)*nnz);
	stat2=cudaMalloc((void**)&row,sizeof(int)*(m+1));
	stat3=cudaMemcpy(crow,c,sizeof(int)*nnz,cudaMemcpyHostToDevice);
	stat6=cusparseCreate(&handle);
	stat5=cusparseXcoo2csr(handle,crow,nnz,m,row,CUSPARSE_INDEX_BASE_ZERO);
	stat4=cudaMemcpy(rrow,row,sizeof(int)*(m+1),cudaMemcpyDeviceToHost);
	stat7=cusparseDestroy(handle);
	if(stat1!=cudaSuccess||
			stat2!=cudaSuccess||
			stat3!=cudaSuccess||
			stat4!=cudaSuccess){
		printf("runtime API error\n");
		exit(-1);
	}
	if(stat5!=CUSPARSE_STATUS_SUCCESS||
			stat6!=CUSPARSE_STATUS_SUCCESS||
			stat7!=CUSPARSE_STATUS_SUCCESS){
		printf("cuSparse API error\n");
		exit(-1);
	}
	cudaFree(row);
	cudaFree(crow);
}
