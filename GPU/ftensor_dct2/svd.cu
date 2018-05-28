#include"svd.h"
#include<cusolverDn.h>
void svd(int m,int n,cuComplex* T,cuComplex* U,cuComplex* V,float* S){
     cusolverDnHandle_t handle;
     gesvdjInfo_t params=NULL;
     int* info=NULL;
     int echo=1;
     int lda=0;
     lda=m;
     int ldu=0;
     ldu=m;
     int ldv=0;
     ldv=n;
     int lwork=0;
     cuComplex* work=NULL;
     float* s=NULL;
     cuComplex* u=NULL;
     cuComplex* v=NULL;
     cuComplex* t=NULL;
     cusolverStatus_t status=CUSOLVER_STATUS_SUCCESS;
     status=cusolverDnCreate(&handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnCreateGesvdjInfo(&params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     cudaError_t stat1=cudaSuccess;
     cudaError_t stat2=cudaSuccess;
     cudaError_t stat3=cudaSuccess;
     cudaError_t stat4=cudaSuccess;
     cudaError_t stat5=cudaSuccess;
     cudaError_t stat6=cudaSuccess;
     stat1=cudaMalloc((void**)&info,sizeof(int));
     int* inf=(int*)malloc(sizeof(int));
     stat2=cudaMalloc((void**)&u,sizeof(cuComplex)*m*((m<n)?m:n));
     stat3=cudaMalloc((void**)&v,sizeof(cuComplex)*n*((m<n)?m:n));
     stat4=cudaMalloc((void**)&s,sizeof(float)*((m<n)?m:n));
     stat5=cudaMalloc((void**)&t,sizeof(cuComplex)*m*n);
     stat6=cudaMemcpy(t,T,sizeof(cuComplex)*m*n,cudaMemcpyHostToDevice);
     if(
    		 stat1!=cudaSuccess||
    		 stat2!=cudaSuccess||
    		 stat3!=cudaSuccess||
    		 stat4!=cudaSuccess||
    		 stat5!=cudaSuccess||
    		 stat6!=cudaSuccess){
    	 printf("cuda malloc error\n");
    	 exit(-1);
     }
     if(cusolverDnCgesvdj_bufferSize(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 m,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 &lwork,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj_bufferSize failed\n");
    	 exit(-1);

     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("synchronize failed");
    	 exit(-1);
     }
     stat1=cudaMalloc((void**)&work,sizeof(cuComplex)*lwork);
     assert(stat1==cudaSuccess);
     if(cusolverDnCgesvdj(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 lda,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 work,
    		 lwork,
    		 info,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj err\n");
    	 return;
     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("cuda synchronize err\n");
    	 return;
     }
     stat1=cudaMemcpy(U,u,sizeof(cuComplex)*m*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(V,v,sizeof(cuComplex)*n*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(S,s,sizeof(float)*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     cudaMemcpy(inf,info,sizeof(int),cudaMemcpyDeviceToHost);
     printf("info %d\n",*inf);
     free(inf);
     stat1=cudaFree(u);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(v);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(s);
     assert(stat1==cudaSuccess);
     cudaFree(info);
     cudaFree(work);
     status=cusolverDnDestroy(handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnDestroyGesvdjInfo(params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     printf("svd success\n");
}

void svd_getv(int m,int n,cuComplex *A,cuComplex *VT,cuComplex* U,float* S){
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;



	int lda=m,ldu=m,ldvt=n;
	cuComplex *d_A;
    cuComplex *d_U;
  	float *d_S;
  	cuComplex *d_VT;
  	int *devInfo=NULL;
  	cuComplex *d_work = NULL;
  	float *r_work = NULL;
  	int lwork =0;

  	int info_gpu = 0;
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	//copy A U S and VT to device
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuComplex)*lda*n);
	cudaStat2 = cudaMalloc((void**)&d_U, sizeof(cuComplex)*ldu*m);
	cudaStat3 = cudaMalloc((void**)&d_S, sizeof(float)*n);
	cudaStat4 = cudaMalloc((void**)&d_VT, sizeof(cuComplex)*ldvt*n);
	cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuComplex)*lda*n, cudaMemcpyHostToDevice);
/*	cudaStat2 = cudaMemcpy(d_U, U, sizeof(cuComplex)*ldu*m, cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(d_S, S, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(d_VT, VT, sizeof(cuComplex)*ldvt*n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);*/

	cusolver_status = cusolverDnCgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuComplex)*lwork);
	cudaStat3 = cudaMalloc((void**)&r_work, sizeof(float)*lwork);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat3);

	//compute SVD
	cusolver_status = cusolverDnCgesvd(
		cusolverH,
		'A',
		'A',
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		ldu,
		d_VT,
		ldvt,
		d_work,
		lwork,
		r_work,
		devInfo
		);

	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	//check if SVD is good or not
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(0 == info_gpu);

	//copy U S and VT to host
	cudaStat1 = cudaMemcpy(U, d_U, sizeof(cuComplex)*ldu*m, cudaMemcpyDeviceToHost);
	 cudaStat2 = cudaMemcpy(S, d_S, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(VT, d_VT, sizeof(cuComplex)*ldvt*n, cudaMemcpyDeviceToHost);
	 assert(cudaSuccess == cudaStat1);
	 assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (d_A) cudaFree(d_A);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	if (r_work) cudaFree(r_work);

	if (cublasH) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);

	cudaDeviceReset();
}
