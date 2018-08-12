#if 0
#include<cuda.h>
#include<malloc.h>
#include"gettime.c"
#include"tensor.c"
#include"transpose.h"
#include<stdio.h>
#include"dct.h"

int main(int argc,char* argv[]){

	int k1,l1;
        int batch;
        if(argc==4){
	k1=atoi(argv[1]);
        l1=atoi(argv[2]);
        batch=atoi(argv[3]);}else{
        fprintf(stderr,"[%s:%d] input error,try again",__FUNCTION__,__LINE__);
        }
   
	double time1,time2,time3;
	time1=gettime();
	
	int threads;
	int blocks;
        
//======================create data============================================	
	float* A3=(float*)malloc(sizeof(float)*batch*k1*l1);
	float* A4=(float*)malloc(sizeof(float)*batch*k1*l1);
	float* A5=(float*)malloc(sizeof(float)*batch*k1*l1*2);
        srand((unsigned)time(NULL));
         for(int i=0;i<k1*l1*batch;i++){
            A3[i]=(float)rand()/(RAND_MAX/100);
          }
    int block=k1*batch;
    const int BLOCK=30000;
    float* swap3;
    float* swap1;
    float* swap4;
    float * E1=(float*)malloc(sizeof(float)*batch*k1*l1);
    float * E2=(float*)malloc(sizeof(float)*batch*k1*l1);
    float * E3=(float*)malloc(sizeof(float)*batch*k1*l1);
    cudaMalloc((void**)&swap1,sizeof(float)*batch*k1*l1);
    cudaMalloc((void**)&swap3,sizeof(float)*batch*k1*l1);
    cudaMalloc((void**)&swap4,sizeof(float)*batch*k1*l1);
    float* swap6;
    float* swap7;
    cudaMalloc((void**)&swap6,sizeof(float)*batch*k1*l1);
    cudaMalloc((void**)&swap7,sizeof(float)*batch*k1*l1);
    time1=gettime();
//=========================dct2================================================
    cudaMemcpy(swap1,A3,sizeof(float)*batch*k1*l1,cudaMemcpyHostToDevice);
    dct_batch<<<BLOCK,THREADS>>>(swap1,l1,k1*batch,swap3);
   // cudaMemcpy(E1,swap3,sizeof(float)*batch*k1*l1,cudaMemcpyDeviceToHost);
   // t_scalar_transpose(l1,k1,batch,E1,E2);
    if(batch*k1*l1<512){
        	threads=k1*l1*batch;
        	blocks=1;
        }else{
        	threads=512;
        	blocks=(batch*k1*l1%512==0)?batch*k1*l1/512:batch*k1*l1/512+1;
        }
    d_batch_transpose<<<blocks,threads>>>(swap4,swap3,k1,l1,batch);
    //cudaMemcpy(E3,swap4,sizeof(float)*batch*k1*l1,cudaMemcpyDeviceToHost); 
    block=l1*batch;
   
   // cudaMemcpy(swap3,E2,sizeof(float)*batch*k1*l1,cudaMemcpyHostToDevice);
    dct_batch<<<BLOCK,THREADS>>>(swap4,k1,l1*batch,swap6);
   // cudaMemcpy(A4,swap6,sizeof(float)*batch*k1*l1,cudaMemcpyDeviceToHost);
    d_batch_transpose<<<blocks,threads>>>(swap7,swap6,l1,k1,batch);
    cudaDeviceSynchronize();
    time2=gettime();
//============================================================================
//======================idct2=================================================
           block=l1*batch;
           d_batch_transpose<<<blocks,threads>>>(swap6,swap7,k1,l1,batch);
	  // cudaMemcpy(swap6,A4,sizeof(float)*batch*k1*l1,cudaMemcpyHostToDevice);
	   idct_batch<<<BLOCK,THREADS>>>(swap6,k1,l1*batch,swap3);
	   //cudaMemcpy(E1,swap3,sizeof(float)*batch*k1*l1,cudaMemcpyDeviceToHost);
           //t_scalar_transpose(l1,k1,batch,E1,E2);
           d_batch_transpose<<<blocks,threads>>>(swap4,swap3,l1,k1,batch);
            block=k1*batch;
	  // cudaMemcpy(swap3,E2,sizeof(float)*batch*k1*l1,cudaMemcpyHostToDevice);
	   idct_batch<<<BLOCK,THREADS>>>(swap4,l1,k1*batch,swap1);
	   cudaMemcpy(A5,swap1,sizeof(float)*batch*k1*l1,cudaMemcpyDeviceToHost);
           time3=gettime();
//===============================================================================
      /*  for(int i=0;i<batch*k1*l1;i++){	
       printf("%f\n",A3[i]-A5[i]);
        }*/
    printf("%d %d %d %f %f\n",k1,l1,batch,time2-time1,time3-time2);
     

  /*  cudaFree(swap3);
    cudaFree(swap1);
    cudaFree(swap4);
    cudaFree(swap6);
    cudaFree(swap7);
    free(A3);
    free(A4); 
    free(A5);
    free(E1);
    free(E2);*/
	return 0;
}
#endif
