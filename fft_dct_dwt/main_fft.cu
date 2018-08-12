/***************************************************
*   test fft 
*   AUTHOR:haili
*
****************************************************/
#if 0
#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include"fft.h"
#include"gettime.c"
#include<time.h>
int main(int argc,char* argv[]){
      int m,n,batch;
      if(argc==4){
        m=atoi(argv[1]);
        n=atoi(argv[2]);
        batch=atoi(argv[3]);
       }else{
           fprintf(stderr,"[%s:%d]input err!try again!\n",__FUNCTION__,__LINE__);
           exit(-1);
           }
 //----------create data------------------------------
      srand((unsigned)time(NULL));
      cuComplex* data=(cuComplex*)malloc(sizeof(cuComplex)*batch*m*n);
      for(int i=0;i<m*n*batch;i++){
            data[i].x=(float)rand()/(RAND_MAX/100);
            data[i].y=(float)rand()/(RAND_MAX/100);
          }
//----------fft2-------------------------------------
//      for(int i=0;i<m*n*batch;i++){
//          printf("%f,%f\n",data[i].x,data[i].y);
//          }
    double time1,time2,time3,time4;
    cuComplex* result=(cuComplex*)malloc(sizeof(cuComplex)*batch*n*m);
    time1=gettime();
    fft2_batch(m,n,data,result,batch);
    time2=gettime();
//      for(int i=0;i<m*n*batch;i++){
//          printf("%f,%f\n",result[i].x,result[i].y);
//          }
    time4=gettime();
    ifft2_batch(m,n,result,data,batch);
    time3=gettime();
    printf("%d %d %d %f %f\n",m,n,batch,time2-time1,time3-time4);
//      for(int i=0;i<m*n*batch;i++){
//          printf("%f,%f\n",data[i].x/6,data[i].y/6);
//          }
return 0;
}
#endif
