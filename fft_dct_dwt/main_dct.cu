//=====================================================================
//  DCT2 BATCH 
//   AUTHOR:HAILI
//======================================================================
#if 0
#include"dct.h"
#include"gettime.c"
#include<cuda_runtime.h>
int main(int argc,char *argv[])
{
    int m;
    int n;
    int batch;
    if(argc==4){
         m=atoi(argv[1]);
         n=atoi(argv[2]);
         batch=atoi(argv[3]);
       }else{
          fprintf(stderr,"[%s:%d]input error,try again!\n",__FUNCTION__,__LINE__);   
           exit(-1);
              }
       float* d_data;
       float* d_result;
       cudaMalloc((void**)&d_data,sizeof(float)*m*n*batch);
       cudaMalloc((void**)&d_result,sizeof(float)*m*n*batch);
       float* data=(float*)malloc(sizeof(float)*n*m*batch);
       float* result=(float*)malloc(sizeof(float)*m*n*batch*2);
       float* data_test=(float*)malloc(sizeof(float)*n*m*batch);
//---------------create data----------------------------------------------
    srand((unsigned(time(NULL))));
    for(int i=0;i<m*n*batch;i++){
          data[i]=i;  
    //data[i]=(float)rand()/(RAND_MAX/100); 
      }
   /* for(int i=0;i<m*n*batch;i++){
       printf("%f\n",data[i]);
      }*/
//--------------------dct2-----------------------------------------------
    double time1,time2,time3,time4;
    cudaMemcpy(d_data,data,sizeof(float)*m*n*batch,cudaMemcpyHostToDevice);
    int blocks=batch;
    int threads=THREADS;
    time1=gettime();
    dct2_batch<<<blocks,threads>>>(d_data,m,n,d_result,batch);
    
   // printf("dsajdslk\n");
    cudaMemcpy(result,d_result,sizeof(float)*batch*m*n,cudaMemcpyDeviceToHost);
    time4=gettime();
    /* for(int i=0;i<m*n*batch;i++){
       printf("%f\n",result[i]);
       }*/

    time2=gettime();
    idct2_batch<<<blocks,threads>>>(d_result,m,n,d_data,batch);
    
    //printf("dhahah\n");
    cudaMemcpy(data_test,d_data,sizeof(float)*batch*m*n,cudaMemcpyDeviceToHost);
    time3=gettime();
     /*for(int i=0;i<m*n*batch;i++){
       printf("%f   %f\n",data_test[i]-data[i],data_test[i]);
         }*/
    printf("%d %d %d %f %f\n",m,n,batch,time4-time1,time3-time2);
}
#endif
