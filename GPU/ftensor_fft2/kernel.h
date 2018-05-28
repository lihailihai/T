/*
 * kernel.h
 *
 *  Created on: Apr 16, 2018
 *      Author: haili
 */

#ifndef KERNEL_H_
#define KERNEL_H_
#include<cuda_runtime.h>
#include"common.h"
typedef struct data_t{
	float a;
	int b;
}cdata;
__device__ void swap(cdata &a,cdata &b);
__global__ void odd_even(float* s,cdata* A,int len,int r);
__global__ void d_tensor_scalar(cuComplex*,cuComplex*,const int,const int,
		const int,const int);
__global__ void d_tensor_scalar_tensor(cuComplex*,cuComplex*,const int,
		const int,const int,const int);
__global__ void d_matview_transpose(cuComplex*,cuComplex*,const int ,const int,
		const int,const int);
__global__ void d_csrtomatview(cuComplex*,cuComplex*,const int *,const int *,
		const int,const int,const int,const int);
__global__ void d_zero(cuComplex*,int len);
#endif /* KERNEL_H_ */
