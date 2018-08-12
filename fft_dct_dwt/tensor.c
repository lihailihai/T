#include"tensor.h"
#include<cuda_runtime.h>
float* t_scalar_transpose(int k,int l,int batch,const float* t,float* T){
    
    	for(int p=0;p<batch;p++){
    		for(int i=0;i<l;i++){
    			for(int j=0;j<k;j++){
    				T[l*k*p+l*j+i]=t[l*k*p+i*k+j];
    			}
    		}
    	}
	return T;
}

		
