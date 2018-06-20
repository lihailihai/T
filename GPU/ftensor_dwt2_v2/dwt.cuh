/*
 * dwt.cuh
 *
 *  Created on: Jun 12, 2018
 *      Author: haili
 */

#ifndef DWT_CUH_
#define DWT_CUH_
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#define THREADS 128
__global__ void dwt_batch(float* A,const int len,const int batch,float* B);
__global__ void idwt_batch(float* A,const int len,const int reclen,const int batch,float* B);
__device__ void exch(int tid,int l,float* A,float* B);

#endif /* DWT_CUH_ */
