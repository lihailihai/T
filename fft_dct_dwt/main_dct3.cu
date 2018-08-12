#include"dct.h"
int main(int argc,char **argv){
	int m;
	int n;
	if(argc==3){
		m=atoi(argv[1]);	
		n=atoi(argv[2]);	
		}else{
		fprintf(stderr,"[%s:%d]input format error,try again!",__FUNCTION__,__LINE__);
		}
//*******************create data************************************
	float* data=(float*)malloc(sizeof(float)*n*m);
	float* out=(float*)malloc(sizeof(float)*m*n);
	for(int i=0;i<m*n;i++){
		data[i]=i;
	}
//******************memery malloc***********************************
	float* d_data=NULL;
	float* d_out=NULL;
	cudaMalloc((void**)&d_data,sizeof(float)*m*n);
	cudaMalloc((void**)&d_out,sizeof(float)*m*n);
//******************dct2*******************************************
	cudaMemcpy(d_data,data,sizeof(float)*m*n,cudaMemcpyHostToDevice);
        dim3 block(32,32,1);
        dim3 grid(m,1,1);
       
	dct2<<<grid,block>>>(d_data,m,n,d_out);
	cudaMemcpy(out,d_out,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	for(int j=0;j<m;j++){
	for(int i=0;i<n;i++){
	printf("%f ",out[j*n+i]);
	}
	printf("\n");
	}
	printf("------------------------------\n");
//****************idct2*******************************************
	cudaMemcpy(d_out,out,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	idct2<<<grid,block>>>(d_out,m,n,d_data);
	cudaMemcpy(out,d_data,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	for(int j=0;j<m;j++){
	for(int i=0;i<n;i++){
	printf("%f ",out[j*n+i]);
	}
	printf("\n");
	}
}
