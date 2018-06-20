
#include<cuda.h>
#include<malloc.h>
#include"gettime.c"
#include"tensor.c"
#include<assert.h>
#include"svd.h"
#include"check.h"
#include"multi_mm.h"
#include"kernel.h"
#include<stdio.h>
#include"dwt.cuh"

int main(int argc,char* argv[]){

	extern void cootocsr(int,int,int*,int*);
	int m1,n1,k1,l1;
	printf("input the size of tensor(m*n*k*l):");
	scanf("%d%d%d%d",&m1,&n1,&k1,&l1);
	printf("m=%d,n=%d,k=%d,l=%d\n",m1,n1,k1,l1);
	int m=m1;
    int n=n1;
    int k=((k1+7)/2)*2;
	int l=((l1+7)/2)*2;
	int k2=((k1+7)/2);
    int l2=((l1+7)/2);
	/*===========read rgbdata==================*/
	printf("data loading………………………\n");
    FILE* f=NULL;
    const char* F_PATH="/home/haili/Documents/MATLAB/kobe.txt";
    f = fopen(F_PATH,"r");
    if(!f){
    	printf("Error when open file\n");
    	exit(1);
    }
	char line[100];
	int num=0;
	float ten;
	cuComplex* tensor=(cuComplex *)malloc(sizeof(cuComplex)*m*n*k1*l1);
	for(int i=0;i<m*n*k1*l1;i++){
	    if(fgets(line,100,f)!=NULL){
		sscanf(line,"%f",&ten);
		tensor[num].x=ten;
		tensor[num].y=0;
		num++;
	}
		}
    printf("count:%d \n",num);
    fclose(f);
	cuComplex* w=(cuComplex*)malloc(m*n*sizeof(cuComplex));
	//store
	float* S=(float*)malloc(sizeof(float)*((m<n)?m:n));
	cuComplex* U=(cuComplex*)malloc(sizeof(cuComplex)*m*m);
	cuComplex* V=(cuComplex*)malloc(sizeof(cuComplex)*n*n);

	float* S1=(float*)malloc(sizeof(float)*((m<n)?m:n)*k*l);
	cuComplex* U1=(cuComplex*)malloc(sizeof(cuComplex)*k*l*m*((m<n)?m:n));
	cuComplex* V1=(cuComplex*)malloc(sizeof(cuComplex)*k*l*n*((m<n)?m:n));

	cuComplex* U2=(cuComplex*)malloc(sizeof(cuComplex)*k*l*m*((m<n)?m:n));
	cuComplex* V2=(cuComplex*)malloc(sizeof(cuComplex)*k*l*n*((m<n)?m:n));


	cufftComplex* A=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* B=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* C=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* D=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* E=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* F=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* G=(cufftComplex*)malloc(m*n*k1*l1*sizeof(cufftComplex));
	if(!w||!A||!C||!D||!S||!U||!V){
		printf("host memory allocation failed");
		exit(-1);
	}
	double time1,time2,time3;
	time1=gettime();
	printf(" shishu tensor\n");
	//w=createtensor(m,n,k,l,w);
//	printtensor(m,n,k,l,w);
	//free(w);
	//create  4D tensor of size m*n*k*l
	/*A=createctensor(m,n,k,l,A);
	printf("cucomplex tensor\n");
    printctensor(m,n,k,l,A);*/
    //transform tensor to tensor_scalar
	int threads;
	int blocks;
	cuComplex* A1;
	cuComplex* A2;
	cuComplex* A6;
	cuComplex* A7;
	float* B1;
	float* B2;
	float* A3=(float*)malloc(sizeof(float)*m*n*k1*l1);
	float* A4=(float*)malloc(sizeof(float)*m*n*k*l);
	float* A5=(float*)malloc(sizeof(float)*m*n*k*l);
	cudaMalloc((void**)&B1,sizeof(float)*n*m*k*l);
	cudaMalloc((void**)&B2,sizeof(float)*n*m*k1*l1);
	cudaMalloc((void**)&A1,sizeof(cuComplex)*n*m*k1*l1);
	cudaMalloc((void**)&A2,sizeof(cuComplex)*n*m*k1*l1);
	cudaMalloc((void**)&A6,sizeof(cuComplex)*n*m*k*l);
	cudaMalloc((void**)&A7,sizeof(cuComplex)*n*m*k*l);
	cudaMemcpy(A1,tensor,sizeof(cuComplex)*m*n*k1*l1,cudaMemcpyHostToDevice);
    if(m*n*k1*l1<512){
    	threads=m*k1*n*l1;
    	blocks=1;
    }else{
    	threads=512;
    	blocks=(m*n*k1*l1%512==0)?m*n*k1*l1/512:m*n*k1*l1/512+1;
    }
    d_tensor_scalar<<<blocks,threads>>>(A2,A1,m,n,k1,l1);
    cudaMemcpy(G,A2,sizeof(cuComplex)*m*n*k1*l1,cudaMemcpyDeviceToHost);
	printf("out put G\n");
	for(int i=0;i<m*n*k1*l1;i++){
			A3[i]=G[i].x;
			}
	//printctensor(m,n,k,l,G);
	//cudaFree(A1);cudaFree(A2);

   // C=tensor_scalar(m,n,k,l,tensor,C);


   /* for(int i=0;i<m*n*k*l;i++){
    printf("%f%s%fi\n",C[i].x-G[i].x,
    		(C[i].y-G[i].y)<0?"":"+",C[i].y-G[i].y);
    }*/
   
   // cudaMemcpy(G,A1,sizeof(cuComplex)*m*n*k*l,cudaMemcpyDeviceToHost);
    printf("test tensor_scalar to tensor\n");
   /* for(int i=0;i<m*n*k*l;i++){
        printf("%f%s%fi\n",A[i].x-G[i].x,
        		(A[i].y-G[i].y)<0?"":"+",A[i].y-G[i].y);
        }*/
    printf("tensor scalar \n");
  //  printctensor(m,n,k,l,C);
    //take  2Ddwt
    int block=l1*m*n;
    float* swap3;
    float* swap1;
    float * E1=(float*)malloc(sizeof(float)*m*n*k*l1);
    float * E2=(float*)malloc(sizeof(float)*m*n*k*l1);
    cudaMalloc((void**)&swap3,sizeof(float)*m*n*k*l1);
    cudaMalloc((void**)&swap1,sizeof(float)*m*n*k1*l1);
    cudaMemcpy(swap1,A3,sizeof(float)*n*m*k1*l1,cudaMemcpyHostToDevice);
    dwt_batch<<<block,THREADS>>>(swap1,k1,l1*m*n,swap3);
    cudaMemcpy(E1,swap3,sizeof(float)*m*n*k*l1,cudaMemcpyDeviceToHost);
    t_scalar_transpose(m,n,k,l1,E1,E2);
   /* for(int i=0;i<m*n*k*l1;i++){
       printf("%f\n",E2[i]);}*/
    block=k*m*n;
    float* swap6;
    cudaMalloc((void**)&swap6,sizeof(float)*m*n*k*l);
    cudaMemcpy(swap3,E2,sizeof(float)*m*n*k*l1,cudaMemcpyHostToDevice);
    dwt_batch<<<block,THREADS>>>(swap3,l1,k*n*m,swap6);
    cudaMemcpy(A4,swap6,sizeof(float)*m*n*k*l,cudaMemcpyDeviceToHost);
    t_scalar_transpose(m,n,l,k,A4,A5);
    for(int d=0;d<k*l*m*n;d++){
        		B[d].x=A5[d];
        		B[d].y=0;
        	}
  //  fft_batch(m,n,k,l,G,B);
    printf("output batch dwt…………………………………………………………………………\n");
  //  printctensor(m,n,k,l,B);
    //transform tensor_scalar to tensor
    //E=tensor_scalartotensor(m,n,k,l,B,E);
    if(m*n*k*l<512){
        	threads=m*k1*n*l1;
        	blocks=1;
        }else{
        	threads=512;
        	blocks=(m*n*k*l%512==0)?m*n*k*l/512:m*n*k*l/512+1;
        }
    cudaMemcpy(A6,B,sizeof(cuComplex)*m*n*k*l,cudaMemcpyHostToDevice);
    d_tensor_scalar_tensor<<<blocks,threads>>>(A7,A6,m,n,k,l);
    printf("tensor scalar to tensor\n");
   // printctensor(m,n,k,l,E);
   // matview_transpose(m,n,k,l,E,F);
   // cudaMemcpy(A2,E,sizeof(cuComplex)*m*n*k*l,cudaMemcpyHostToDevice);
    d_matview_transpose<<<blocks,threads>>>(A6,A7,m,n,k,l);
    cudaMemcpy(F,A6,sizeof(cuComplex)*m*n*k*l,cudaMemcpyDeviceToHost);
    printf("test matviw_transpose………………………………………………\n");
 /*   for(int i=0;i<m*n*k*l;i++){
               printf("%f%s%fi\n",F[i].x-G[i].x,
               		(F[i].y-G[i].y<0)?"":"+",F[i].y-G[i].y);
               }
        for(int i=0;i<m*n*k*l;i++){
            printf("%f%s%fi\n",G[i].x,
            		G[i].y<0?"":"+",G[i].y);
            }*/
    printf("……………………………………………………………………………………………………\n");
   // printctensor(m,n,k,l,F);
    //take SVD
    cufftComplex** pmm=(cufftComplex**)F;
    cufftComplex (*pm)[m*n]=(cufftComplex (*)[m*n])pmm;
    cufftComplex* temp=(cufftComplex*)malloc(sizeof(cufftComplex)*m*n);
    for(int q=0;q<k*l;q++){
    for(int i=0;i<m*n;i++){
    	temp[i]=pm[q][i];
    }
   /* for(int i=0;i<m*n;i++){
       	printf("temp %f %f\n",temp[i].x,temp[i].y);
       }*/
    svd(m,n,(cuComplex*)temp,U,V,S);

    for(int j=0;j<m*((m<n)?m:n);j++){

    	 U2[j+q*m*((m<n)?m:n)].x=U[j].x;
    	 U2[j+q*m*((m<n)?m:n)].y=U[j].y;

     }

    for(int j=0;j<n*((m<n)?m:n);j++){

        V2[j+q*n*((m<n)?m:n)].x=V[j].x;
        V2[j+q*n*((m<n)?m:n)].y=V[j].y;
    }

   for(int j=0;j<((m<n)?m:n);j++){
    	S1[j+q*((m<n)?m:n)]=S[j];
    }

    }
    //output S1 U1 V1
  //  printSVD(m,n,k,l,S1,U2,V2);
    matview_transpose(((m<n)?m:n),m,k,l,U2,U1);
    //U1 to coo
    int count=0;
    for(int i=0;i<m*((m<n)?m:n)*k*l;i++){
    if(U1[i].x!=0||U1[i].y!=0){
     count++;
    }}
    printf("output U1 count %d\n",count);
    int* row_array=(int*)malloc(sizeof(int)*count);
    int* col_array=(int*)malloc(sizeof(int)*count);
    cuComplex* data_array=(cuComplex*)malloc(sizeof(cuComplex)*count);
    coo* t1coo=(coo*)malloc(sizeof(coo));
    coo* tcoo=(coo*)malloc(sizeof(coo));
    tcoo=matviewtocoo(m,((m<n)?m:n),k,l,t1coo,U1,row_array,col_array,data_array);
   /* for(int i=0;i<count;i++){
    	printf("%d %d %f%s%fi\n",tcoo->row_array[i],
    			tcoo->col_array[i],tcoo->data_array[i].x,
    			((tcoo->data_array[i].y<0)?"":"+"),tcoo->data_array[i].y);
    }*/


    //coo to csr
    csr* tcsr_U=(csr*)malloc(sizeof(csr));
    int* rrow=(int*)malloc(sizeof(int)*(m*k*l+1));
   /* for(int i=0;i<count;i++){
    printf("%d\n",tcoo->row_array[i]);}*/
    cootocsr((m*k*l),count,tcoo->row_array,rrow);
    tcsr_U->col_array=NULL;
    tcsr_U->data_array=NULL;
    tcsr_U->row_array=rrow;
    printf("output  U2 csr->row_array\n");
    /*for(int i=0;i<(m*k*l+1);i++){
   	printf("%d\n",tcsr_U->row_array[i]);
    }*/



    count=0;
    //V1 to coo
  /*  matview_transpose(((m<n)?m:n),n,k,l,V2,V1);
    free(V2);
    for(int i=0;i<n*((m<n)?m:n)*k*l;i++){
        if(V1[i].x!=0||V1[i].y!=0){
         count++;
        }}*/
    for(int i=0;i<n*((m<n)?m:n)*k*l;i++){
            if(V2[i].x!=0||V2[i].y!=0){
             count++;}}
    printf("output V1 count %d\n",count);
    int* row_array1=(int*)malloc(sizeof(int)*count);
    int* col_array1=(int*)malloc(sizeof(int)*count);
    cuComplex* data_array1=(cuComplex*)malloc(sizeof(cuComplex)*count);
    coo* tcoo2=(coo*)malloc(sizeof(coo));
    coo* tcoo3=(coo*)malloc(sizeof(coo));
    tcoo3=matviewtocoo(((m<n)?m:n),n,k,l,tcoo3,V2,row_array1,col_array1,data_array1);
   /* for(int i=0;i<count;i++){
        	printf("%d %d %f%s%fi\n",tcoo3->row_array[i],
        			tcoo3->col_array[i],tcoo3->data_array[i].x,
        			((tcoo3->data_array[i].y<0)?"":"+"),tcoo3->data_array[i].y);
        }*/
//V coo to csr
    csr* tcsr_V=(csr*)malloc(sizeof(csr));
    int* rrow1=(int*)malloc(sizeof(int)*(((m<n)?m:n)*k*l+1));
       cootocsr(((m<n)?m:n)*k*l,count,tcoo3->row_array,rrow1);
       tcsr_V->row_array=rrow1;
       printf("output  V2 csr->row_array\n");
     /*  for(int i=0;i<(((m<n)?m:n)*k*l+1);i++){
       	printf("%d\n",tcsr_V->row_array[i]);
       }*/


    count=0;
    //S1 to coo

    for(int i=0;i<((m<n)?m:n)*k*l;i++){
                if(S1[i]!=0){
                 count++;
               //  printf("qian %f\n",S1[i]);
                }
        }
    /*====================================================================set r=====================================================*/
    printf("output S1 count %d\n",count);
 //   cdata* pg;
   // int r=0;
    int r=((m<n)?m:n)*k*l*9/10;
//    float* d_S1=NULL;
    cdata* pg1=(cdata*)malloc(sizeof(cdata)*((m<n)?m:n)*k*l);
    for(int i=0;i<((m<n)?m:n)*k*l;i++){
        pg1[i].a=S1[i];
        pg1[i].b=i;
      }
      cdata swap;
     for(int i=0;i<((m<n)?m:n)*k*l;i++){
        for(int j=i+1;j<((m<n)?m:n)*k*l;j++){
        if(pg1[i].a>pg1[j].a){
         swap.a=pg1[i].a;
         swap.b=pg1[i].b;
         pg1[i].a=pg1[j].a;
         pg1[i].b=pg1[j].b;
         pg1[j].a=swap.a;
         pg1[j].b=swap.b;
       }
      } 
     }
     for(int i=0;i<r;i++){
      S1[pg1[i].b]=0;
     }
   /* for(int i=0;i<((m<n)?m:n)*k*l;i++){
        pg1[i].a=S1[i];
        pg1[i].b=i;
      }
    cudaMalloc((void**)&d_S1,sizeof(float)*((m<n)?m:n)*k*l);
    cudaMalloc((void**)&pg,sizeof(cdata)*((m<n)?m:n)*k*l);
    cudaMemcpy(d_S1,S1,sizeof(float)*((m<n)?m:n)*k*l,cudaMemcpyHostToDevice);
    cudaMemcpy(pg,pg1,sizeof(float)*((m<n)?m:n)*k*l,cudaMemcpyHostToDevice);
    if(m*n*k*l<512){
        	threads=m*k*n*l;
        	blocks=1;
        }else{
        	threads=512;
        	blocks=((((m<n)?m:n)*k*l)%threads==0)?((m<n)?m:n)*k*l/threads:(((m<n)?m:n)*k*l/threads+1);
        }
    odd_even<<<blocks,threads>>>(d_S1,pg,((m<n)?m:n)*k*l,r);
    cudaMemcpy(S1,d_S1,sizeof(float)*((m<n)?m:n)*k*l,cudaMemcpyDeviceToHost);
    cudaMemcpy(pg1,pg,sizeof(cdata)*((m<n)?m:n)*k*l,cudaMemcpyDeviceToHost);*/
    
    count=0;
    for(int i=0;i<((m<n)?m:n)*k*l;i++){
            if(S1[i]!=0){
             count++;

            }
           // printf("hou %f   %f,%d\n",S1[i],pg1[i].a,pg1[i].b);
    }

    printf("output S1 count %d\n",count);
    int* row_array2=(int*)malloc(sizeof(int)*count);
    int* col_array2=(int*)malloc(sizeof(int)*count);
    cuComplex* data_array2=(cuComplex*)malloc(sizeof(cuComplex)*count);
    coo* tcoo4=(coo*)malloc(sizeof(coo));
    for(int i=0,j=0;i<((m<n)?m:n)*k*l;i++){
    	if(S1[i]!=0){
    	row_array2[j]=i;
    	col_array2[j]=i;
    	data_array2[j].x=S1[i];
    	data_array2[j].y=0.000000;
    	j++;
    	}
    }
    tcoo4->row_array=row_array2;
    tcoo4->col_array=col_array2;
    tcoo4->data_array=data_array2;
   /* for(int i=0;i<count;i++){
            	printf("%d %d %f\n",tcoo4->row_array[i],
            			tcoo4->col_array[i],tcoo4->data_array[i].x);
            }*/


    csr* tcsr_S=(csr *)malloc(sizeof(csr));
    tcsr_S->row_array=(int *)malloc(sizeof(int)*(((m<n)?m:n)*k*l+1));
    cootocsr(((m<n)?m:n)*k*l,count,tcoo4->row_array,tcsr_S->row_array);
    count=0;
    /*for(int i=0;i<(((m<n)?m:n)*k*l+1);i++){

    	tcsr_S->row_array[i]=i;
    }*/

  /*  for(int i=0;i<(((m<n)?m:n)*k*l+1);i++){
	   printf("%d\n",tcsr_S->row_array[i]);}*/
    //multi-mm
    //U * S
    cusparseOperation_t transA=CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB=CUSPARSE_OPERATION_NON_TRANSPOSE;
    int min=((m<n)?m:n);
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
        nnzA=tcsr_U->row_array[m*k*l]-tcsr_U->row_array[0];
        nnzB=tcsr_S->row_array[min*k*l]-tcsr_S->row_array[0];

    	stat1=cudaMalloc((void**)&d_A_row_array,sizeof(int)*(m*k*l+1));

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
    		return 0;
    	}

    	if(cudaMemcpy(
    			d_A_row_array,
    			tcsr_U->row_array,
    			sizeof(int)*(m*l*k+1),
    			cudaMemcpyHostToDevice)!=cudaSuccess){
    		printf("cuda memcpy err 1\n");
    		exit(-1);
    	}
    	if(cudaMemcpy(
    			d_A_col_array,
    			tcoo->col_array,
    			sizeof(int)*nnzA,
    			cudaMemcpyHostToDevice)!=cudaSuccess){
    		printf("cuda memcpy err 2\n");
    		exit(-1);
    	}
    	if(cudaMemcpy(
    			d_A_data_array,
    			tcoo->data_array,
    			sizeof(cuComplex)*nnzA,
    			cudaMemcpyHostToDevice)!=cudaSuccess){
    		printf("cuda memcpy err 3\n");
    		exit(-1);
    	}
        if(cudaMemcpy(
        		d_B_row_array,
        		tcsr_S->row_array,
        		sizeof(int)*(min*k*l+1),
        		cudaMemcpyHostToDevice)!=cudaSuccess){
        	printf("cuda memcpy err 4\n");
        	exit(-1);
        }
        if(cudaMemcpy(
        		d_B_col_array,
        		tcoo4->col_array,
        		sizeof(int)*nnzB,
        		cudaMemcpyHostToDevice)!=cudaSuccess){
        	printf("cuda memcpy err 5\n");
        	exit(-1);
        }
        if(cudaMemcpy(
        		d_B_data_array,
        		tcoo4->data_array,
        		sizeof(cuComplex)*nnzB,
        		cudaMemcpyHostToDevice)!=cudaSuccess){
        	printf("cuda memcpy err 6\n");
        	exit(-1);
        }

        cusparseHandle_t handle;
        if(cusparseCreate(&handle)!=CUSPARSE_STATUS_SUCCESS){
        	printf("cuaparsecreate handle failed\n");
        	return 0;
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
        	return 0;
        }
        if(cusparseXcsrgemmNnz(
        		handle,
        		transA,
        		transB,
        		m*k*l,
        		min*k*l,
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
        	return 0;
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
        printf("nnzC %d\n",nnzC);
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
        		min*k*l,
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


       /* printf("output C_row_array\n");
          for(int i=0;i<(m*k*l+1);i++){
       	   printf("%d\n",C_row_array[i]);

          }
          printf("nnzc%d\n",nnzC);
          for(int i=0;i<nnzC;i++){
       	   printf("%d %f%s%fi\n",C_col_array[i],C_data_array[i].x,
       			  ((C_data_array[i].y<0)?"":"+"),C_data_array[i].y);

          }*/

   /*multi_mm(tcsr_U->row_array,tcoo->col_array,tcoo->data_array,
    		tcsr_S->row_array,
    		tcoo4->col_array,
    		tcoo4->data_array,
    		m,((m<n)?m:n),((m<n)?m:n),k,l,
    		CUSPARSE_OPERATION_NON_TRANSPOSE,
    		CUSPARSE_OPERATION_NON_TRANSPOSE
    		);*/
   // *** * V
          cusparseOperation_t transA1=CUSPARSE_OPERATION_NON_TRANSPOSE;
          cusparseOperation_t transB1=CUSPARSE_OPERATION_NON_TRANSPOSE;
                nnzB=tcsr_V->row_array[((m<n)?m:n)*k*l]-tcsr_V->row_array[0];
                cuComplex* d_A_data_array1;
                    	cuComplex* d_B_data_array1;
                    	cuComplex* d_C_data_array1;
                    	int* d_A_row_array1;
                    	int* d_A_col_array1;
                    	int* d_B_row_array1;
                    	int* d_B_col_array1;
                    	int* d_C_row_array1;
                    	int* d_C_col_array1;
            	stat1=cudaMalloc((void**)&d_A_row_array1,sizeof(int)*(m*k*l+1));
            	printf("lihai");
            	stat2=cudaMalloc((void**)&d_A_col_array1,sizeof(int)*nnzC);

            	stat3=cudaMalloc((void**)&d_A_data_array1,sizeof(cuComplex)*nnzC);

            	stat4=cudaMalloc((void**)&d_B_row_array1,sizeof(int)*(((m<n)?m:n)*k*l+1));
            	stat5=cudaMalloc((void**)&d_B_col_array1,sizeof(int)*nnzB);
            	stat6=cudaMalloc((void**)&d_B_data_array1,sizeof(cuComplex)*nnzB);
            	if(
            			stat1!=cudaSuccess||
            			stat2!=cudaSuccess||
            			stat3!=cudaSuccess||
            			stat4!=cudaSuccess||
            			stat5!=cudaSuccess||
            			stat6!=cudaSuccess){
            		printf("cuda malloc faild\n");
            		return 0;
            	}

            	if(cudaMemcpy(
            			d_A_row_array1,
            			C_row_array,
            			sizeof(int)*(m*l*k+1),
            			cudaMemcpyHostToDevice)!=cudaSuccess){
            		printf("cuda memcpy err 1\n");
            		exit(-1);
            	}
            	if(cudaMemcpy(
            			d_A_col_array1,
            			C_col_array,
            			sizeof(int)*nnzC,
            			cudaMemcpyHostToDevice)!=cudaSuccess){
            		printf("cuda memcpy err 2\n");
            		exit(-1);
            	}
            	if(cudaMemcpy(
            			d_A_data_array1,
            			C_data_array,
            			sizeof(cuComplex)*nnzC,
            			cudaMemcpyHostToDevice)!=cudaSuccess){
            		printf("cuda memcpy err 3\n");
            		exit(-1);
            	}
                if(cudaMemcpy(
                		d_B_row_array1,
                		tcsr_V->row_array,
                		sizeof(int)*(((m<n)?m:n)*k*l+1),
                		cudaMemcpyHostToDevice)!=cudaSuccess){
                	printf("cuda memcpy err 4\n");
                	exit(-1);
                }
                if(cudaMemcpy(
                		d_B_col_array1,
                		tcoo3->col_array,
                		sizeof(int)*nnzB,
                		cudaMemcpyHostToDevice)!=cudaSuccess){
                	printf("cuda memcpy err 5\n");
                	exit(-1);
                }
                if(cudaMemcpy(
                		d_B_data_array1,
                		tcoo3->data_array,
                		sizeof(cuComplex)*nnzB,
                		cudaMemcpyHostToDevice)!=cudaSuccess){
                	printf("cuda memcpy err 6\n");
                	exit(-1);
                }
                cusparseHandle_t handle1;
                if(cusparseCreate(&handle1)!=CUSPARSE_STATUS_SUCCESS){
                	printf("cuaparsecreate handle failed\n");
                	return 0;
                }

                cusparseMatDescr_t descrA1;
                cusparseMatDescr_t descrB1;
                cusparseMatDescr_t descrC1;
                status=cusparseCreateMatDescr(&descrA1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseCreateMatDescr(&descrB1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseCreateMatDescr(&descrC1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseSetMatType(descrA1,CUSPARSE_MATRIX_TYPE_GENERAL);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseSetMatType(descrB1,CUSPARSE_MATRIX_TYPE_GENERAL);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseSetMatType(descrC1,CUSPARSE_MATRIX_TYPE_GENERAL);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                int nnzD=0;int baseD=0;
                int* nnzTotalDevHostPtr1=&nnzD;
                if(cudaMalloc((void**)&d_C_row_array1,sizeof(int)*(m*k*l+1))!=cudaSuccess){
                	printf("cuda malloc error\n");
                	return 0;
                }
                if(cusparseXcsrgemmNnz(
                		handle1,
                		transA1,
                		transB1,
                		m*k*l,
                		n*k*l,
                		min*k*l,
                		descrA1,
                		nnzC,
                		d_A_row_array1,
                		d_A_col_array1,
                		descrB1,
                		nnzB,
                		d_B_row_array1,
                		d_B_col_array1,
                		descrC1,
                		d_C_row_array1,
                		nnzTotalDevHostPtr1
                		)!=CUSPARSE_STATUS_SUCCESS){
                	printf("gemmnz error\n");
                	exit(-1);
                }
                if(cudaDeviceSynchronize()!=cudaSuccess){
                	printf("synchronize error\n");
                	return 0;
                }
                if(NULL!=nnzTotalDevHostPtr){
                	nnzD=*nnzTotalDevHostPtr1;
                }
                else{
                	cudaMemcpy(
                			&nnzD,
                			d_C_row_array1+m*k*l,
                			sizeof(int),
                			cudaMemcpyDeviceToHost);
                	cudaMemcpy(
                			&baseD,
                			d_C_row_array1,
                			sizeof(int),
                			cudaMemcpyDeviceToHost);
                	nnzD=-baseD;
                }
                printf("nnzD %d\n",nnzD);
              int*  D_row_array=(int*)malloc(sizeof(int)*(m*k*l+1));
              int * D_col_array=(int*)malloc(sizeof(int)*nnzD);
              cuComplex*  D_data_array=(cuComplex*)malloc(sizeof(cuComplex)*nnzD);
                if(
                		!D_row_array||
                		!D_col_array||
                		!D_data_array){
                	printf("multi_mm malloc error");
                }

                status2=cudaMalloc((void**)&d_C_col_array1,sizeof(int)*nnzD);
                assert(status2==cudaSuccess);
                status2=cudaMalloc((void**)&d_C_data_array1,sizeof(cuComplex)*nnzD);
                assert(status2==cudaSuccess);
                if(cusparseCcsrgemm(
                		handle1,
                		transA1,
                		transB1,
                		m*k*l,
                		n*k*l,
                		min*k*l,
                		descrA1,
                		nnzC,
                		d_A_data_array1,
                		d_A_row_array1,
                		d_A_col_array1,
                		descrB1,
                		nnzB,
                		d_B_data_array1,
                		d_B_row_array1,
                		d_B_col_array1,
                		descrC1,
                		d_C_data_array1,
                		d_C_row_array1,
                		d_C_col_array1
                		)!=CUSPARSE_STATUS_SUCCESS){
                	printf("csrgemm error\n");
                	exit(-1);
                }
                status2=cudaDeviceSynchronize();
                assert(status2==cudaSuccess);
                status2=cudaMemcpy(
                		D_row_array,
                		d_C_row_array1,
                		sizeof(int)*(m*k*l+1),
                		cudaMemcpyDeviceToHost);
                assert(status2==cudaSuccess);
                status2=cudaMemcpy(
                		D_col_array,
                		d_C_col_array1,
                		sizeof(int)*nnzD,
                		cudaMemcpyDeviceToHost);
                assert(status2==cudaSuccess);
                status2=cudaMemcpy(
                		D_data_array,
                		d_C_data_array1,
                		sizeof(cuComplex)*nnzD,
                		cudaMemcpyDeviceToHost);
                assert(status2==cudaSuccess);
                status=cusparseDestroyMatDescr(descrA1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseDestroyMatDescr(descrB1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseDestroyMatDescr(descrC1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status=cusparseDestroy(handle1);
                assert(status==CUSPARSE_STATUS_SUCCESS);
                status2=cudaFree(d_A_row_array1);
                assert(status2==cudaSuccess);
                status2=cudaFree(d_A_col_array1);
                assert(status2==cudaSuccess);
                status2=cudaFree(d_A_data_array1);
                assert(status2==cudaSuccess);
                status2=cudaFree(d_B_row_array1);
                assert(status2==cudaSuccess);
                status2=cudaFree(d_B_col_array1);
                assert(status2==cudaSuccess);
                status2=cudaFree(d_B_data_array1);
                assert(status2==cudaSuccess);
                if(m*k*l*n<512){
                	threads=m*k*l*n;
                	blocks=1;
                }else{
                	threads=512;
                	blocks=((m*k*l*n)%threads==0)?(m*k*l*n/threads):(m*k*n*l/threads+1);
                }
                d_zero<<<blocks,threads>>>(A6,m*n*k*l);
                if(m*k*l<512){
                                	threads=m*k*l;
                                	blocks=1;
                                }else{
                                	threads=512;
                                	blocks=((m*k*l)%threads==0)?(m*k*l/threads):(m*k*l/threads+1);
                                }
                d_csrtomatview<<<blocks,threads>>>(A6,d_C_data_array1,d_C_row_array1,d_C_col_array1,m,n,k,l);
                cudaMemcpy(B,A6,sizeof(cuComplex)*m*n*k*l,cudaMemcpyDeviceToHost);
                printf("test csr to matview()\n");
               // printctensor(m,n,k,l,G);
                cudaFree(d_C_col_array1);
                cudaFree(d_C_row_array1);
                cudaFree(d_C_data_array1);
                printf("output D_row_array\n");
                        /* for(int i=0;i<(m*k*l+1);i++){
                      	   printf("%d\n",D_row_array[i]);

                         }*/
                         printf("nnzD%d\n",nnzD);
                        /* for(int i=0;i<nnzD;i++){
                      	   printf("%d %f%s%fi\n",D_col_array[i],(D_data_array[i].x-G[i].x),
                      			  (((D_data_array[i].y)-G[i].y<0)?"":"+"),
		                        (D_data_array[i].y-G[i].y));

                         }*/
    //take iDWT2
    C=tensor_scalar(m,n,k,l,B,C);
   // printctensor(m,n,k,l,C);
    for(int i=0;i<m*n*k*l;i++){
    			A5[i]=C[i].x;
    			}
    t_scalar_transpose(m,n,k,l,A5,A4);
       block=k*m*n;
	   cudaMemcpy(swap6,A4,sizeof(float)*m*n*k*l,cudaMemcpyHostToDevice);
	   idwt_batch<<<block,THREADS>>>(swap6,l2,l1,k*m*n,swap3);
	   cudaMemcpy(E1,swap3,sizeof(float)*m*n*k*l1,cudaMemcpyDeviceToHost);
   t_scalar_transpose(m,n,l1,k,E1,E2);
       block=l1*m*n;
	   cudaMemcpy(swap3,E2,sizeof(float)*m*n*k*l1,cudaMemcpyHostToDevice);
	   idwt_batch<<<block,THREADS>>>(swap3,k2,k1,l1*m*n,swap1);
	   cudaMemcpy(A3,swap1,sizeof(float)*m*n*k1*l1,cudaMemcpyDeviceToHost);

    for(int i=0;i<m*n*k1*l1;i++){
       			tensor[i].x=A3[i];
       			tensor[i].y=0;
       			}
  //  ifft_batch(m,n,k,l,C,B);
    G=tensor_scalartotensor(m,n,k1,l1,tensor,G);
    printf("test result\n");
   // printctensor(m,n,k,l,E);
   /* for(int i=0;i<m*n*k*l;i++){
    	printf("%f %f \n",E[i].x/(k*l)-tensor[i].x,E[i].y/(k*l)-tensor[i].y);
    }*/
    FILE* fp=fopen("/home/haili/Documents/MATLAB/gpu_based_dwt2/compress_result9_10.txt","w");
    if(!fp){
    	printf("open compress_result.txt error\n");
    	exit(-1);
    }
    for(int i=0;i<m*n*k1*l1;i++){
    	fprintf(fp,"%f\n",G[i].x);
    }
    fclose(fp);
   // printctensor(m,n,k,l,D);
   // fft(m,n,k,l,A,B);
	time2=gettime();
	time3=time2-time1;
	printf("time:%.6f\n",time3);
  //  checkkernel();
	free(A);
	free(E);
	free(B);
	free(D);
	free(C);
	free(S);
	free(U);
	free(V);
	free(S1);
	free(U1);
	free(V1);
	free(tcoo);
	free(row_array);
	free(col_array);
	return 0;
}
