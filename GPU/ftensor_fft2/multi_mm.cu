#include"multi_mm.h"
void multi_mm(
		int* A_row_array,
		int* A_col_array,
		cuComplex* A_data_array,
		int* B_row_array,
		int* B_col_array,
		cuComplex* B_data_array,
		int m,
		int n,
		int min,
		int k,
		int l,
		cusparseOperation_t transA,
		cusparseOperation_t transB){
	int nnzA=0,nnzB=0,nnzC=0,baseC=0;
	int* C_row_array;
	int* C_col_array;
	cuComplex* C_data_array;
	cuComplex* d_A_data_array;
	cuComplex* d_B_data_array;
	cuComplex* d_C_data_array;
	int* d_A_row_array;
	int* d_A_col_array;
	int* d_B_row_array;
	int* d_B_col_array;
	int* d_C_row_array;
	int* d_C_col_array;
    cudaError_t stat1=cudaSuccess;
    cudaError_t stat2=cudaSuccess;
    cudaError_t stat3=cudaSuccess;
    cudaError_t stat4=cudaSuccess;
    cudaError_t stat5=cudaSuccess;
    cudaError_t stat6=cudaSuccess;
    nnzA=A_row_array[m*k*l]-A_row_array[0];
    nnzB=B_row_array[min*k*l]-B_row_array[0];

    printf("%d\n%d %d %d %d %d\n",nnzA,m,n,k,l,nnzB);

	stat1=cudaMalloc((void**)&d_A_row_array,sizeof(int)*(m*k*l+1));
	printf("lihai");
	for(int i=0;i<(((m<n)?m:n)*k*l+1);i++){
			    	   printf("%d\n",B_row_array[i]);}
	stat2=cudaMalloc((void**)&d_A_col_array,sizeof(int)*nnzA);

	stat3=cudaMalloc((void**)&d_A_data_array,sizeof(cuComplex)*nnzA);

	stat4=cudaMalloc((void**)&d_B_row_array,sizeof(int)*(min*k*l+1));
	stat5=cudaMalloc((void**)&d_B_col_array,sizeof(int)*nnzB);
	stat6=cudaMalloc((void**)&d_B_data_array,sizeof(cuComplex)*nnzB);
	if(
			stat1!=cudaSuccess||
			stat2!=cudaSuccess||
			stat3!=cudaSuccess||
			stat4!=cudaSuccess||
			stat5!=cudaSuccess||
			stat6!=cudaSuccess){
		printf("cuda malloc faild\n");
		return;
	}

	if(cudaMemcpy(
			d_A_row_array,
			A_row_array,
			sizeof(int)*(m*l*k+1),
			cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("cuda memcpy err 1\n");
		exit(-1);
	}
	if(cudaMemcpy(
			d_A_col_array,
			A_col_array,
			sizeof(int)*nnzA,
			cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("cuda memcpy err 2\n");
		exit(-1);
	}
	if(cudaMemcpy(
			d_A_data_array,
			A_data_array,
			sizeof(cuComplex)*nnzA,
			cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("cuda memcpy err 3\n");
		exit(-1);
	}
    if(cudaMemcpy(
    		d_B_row_array,
    		B_row_array,
    		sizeof(int)*(min*k*l+1),
    		cudaMemcpyHostToDevice)!=cudaSuccess){
    	printf("cuda memcpy err 4\n");
    	exit(-1);
    }
    if(cudaMemcpy(
    		d_B_col_array,
    		B_col_array,
    		sizeof(int)*nnzB,
    		cudaMemcpyHostToDevice)!=cudaSuccess){
    	printf("cuda memcpy err 5\n");
    	exit(-1);
    }
    if(cudaMemcpy(
    		d_B_data_array,
    		B_data_array,
    		sizeof(cuComplex)*nnzB,
    		cudaMemcpyHostToDevice)!=cudaSuccess){
    	printf("cuda memcpy err 6\n");
    	exit(-1);
    }

    cusparseHandle_t handle;
    if(cusparseCreate(&handle)!=CUSPARSE_STATUS_SUCCESS){
    	printf("cuaparsecreate handle failed\n");
    	return;
    }
    cusparseMatDescr_t descrA;
    cusparseMatDescr_t descrB;
    cusparseMatDescr_t descrC;
    cusparseStatus_t status=CUSPARSE_STATUS_SUCCESS;
    status=cusparseCreateMatDescr(&descrA);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseCreateMatDescr(&descrB);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseCreateMatDescr(&descrC);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    int* nnzTotalDevHostPtr=&nnzC;
    if(cudaMalloc((void**)&d_C_row_array,sizeof(int)*(m*k*l+1))!=cudaSuccess){
    	printf("cuda malloc error\n");
    	return;
    }
    if(cusparseXcsrgemmNnz(
    		handle,
    		transA,
    		transB,
    		m*k*l,
    		n*k*l,
    		min*k*l,
    		descrA,
    		nnzA,
    		d_A_row_array,
    		d_A_col_array,
    		descrB,
    		nnzB,
    		d_B_row_array,
    		d_B_col_array,
    		descrC,
    		d_C_row_array,
    		nnzTotalDevHostPtr
    		)!=CUSPARSE_STATUS_SUCCESS){
    	printf("gemmnz error\n");
    	exit(-1);
    }
    if(cudaDeviceSynchronize()!=cudaSuccess){
    	printf("synchronize error\n");
    	return;
    }
    if(NULL!=nnzTotalDevHostPtr){
    	nnzC=*nnzTotalDevHostPtr;
    }
    else{
    	cudaMemcpy(
    			&nnzC,
    			d_C_row_array+m*k*l,
    			sizeof(int),
    			cudaMemcpyDeviceToHost);
    	cudaMemcpy(
    			&baseC,
    			d_C_row_array,
    			sizeof(int),
    			cudaMemcpyDeviceToHost);
    	nnzC=-baseC;
    }

    C_row_array=(int*)malloc(sizeof(int)*(m*k*l+1));
    C_col_array=(int*)malloc(sizeof(int)*nnzC);
    C_data_array=(cuComplex*)malloc(sizeof(cuComplex)*nnzC);
    if(
    		!C_row_array||
    		!C_col_array||
    		!C_data_array){
    	printf("multi_mm malloc error");
    }
    cudaError_t status2=cudaSuccess;
    status2=cudaMalloc((void**)&d_C_col_array,sizeof(int)*nnzC);
    assert(status2==cudaSuccess);
    status2=cudaMalloc((void**)&d_C_data_array,sizeof(cuComplex)*nnzC);
    assert(status2==cudaSuccess);
    if(cusparseCcsrgemm(
    		handle,
    		transA,
    		transB,
    		m*k*l,
    		n*k*l,
    		min*k*l,
    		descrA,
    		nnzA,
    		d_A_data_array,
    		d_A_row_array,
    		d_A_col_array,
    		descrB,
    		nnzB,
    		d_B_data_array,
    		d_B_row_array,
    		d_B_col_array,
    		descrC,
    		d_C_data_array,
    		d_C_row_array,
    		d_C_col_array
    		)!=CUSPARSE_STATUS_SUCCESS){
    	printf("csrgemm error\n");
    	exit(-1);
    }
    status2=cudaDeviceSynchronize();
    assert(status2==cudaSuccess);
    status2=cudaMemcpy(
    		C_row_array,
    		d_C_row_array,
    		sizeof(int)*(m*k*l+1),
    		cudaMemcpyDeviceToHost);
    assert(status2==cudaSuccess);
    status2=cudaMemcpy(
    		C_col_array,
    		d_C_col_array,
    		sizeof(int)*nnzC,
    		cudaMemcpyDeviceToHost);
    assert(status2==cudaSuccess);
    status2=cudaMemcpy(
    		C_data_array,
    		d_C_data_array,
    		sizeof(cuComplex)*nnzC,
    		cudaMemcpyDeviceToHost);
    assert(status2==cudaSuccess);
    status=cusparseDestroyMatDescr(descrA);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseDestroyMatDescr(descrB);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseDestroyMatDescr(descrC);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status=cusparseDestroy(handle);
    assert(status==CUSPARSE_STATUS_SUCCESS);
    status2=cudaFree(d_A_row_array);
    assert(status2==cudaSuccess);
    status2=cudaFree(d_A_col_array);
    assert(status2==cudaSuccess);
    status2=cudaFree(d_A_data_array);
    assert(status2==cudaSuccess);
    status2=cudaFree(d_B_row_array);
    assert(status2==cudaSuccess);
    status2=cudaFree(d_B_col_array);
    assert(status2==cudaSuccess);
    status2=cudaFree(d_B_data_array);
    assert(status2==cudaSuccess);
    cudaFree(d_C_col_array);
    cudaFree(d_C_row_array);
    cudaFree(d_C_data_array);
}




