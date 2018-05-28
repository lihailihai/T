/*
 * dct.cuh
 *
 *  Created on: May 23, 2018
 *      Author: haili
 */

#ifndef DCT_CUH_
#define DCT_CUH_
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define pi 3.14159265358979323846
#define THREADX 32
#define THREADY 16
#define THREADS 256
#define BLOCKS  540*302
__global__ void dct_batch(float*,const int,const int,const int,float*);
__global__ void dct2_batch(float*,const int,const int,const int,float*);
__global__ void dct2_batch2(float*,const int,const int,float*);
__global__ void idct2_batch2(float*,const int,const int,float*);
__global__ void p(float*,const int ,const int ,const int,float*);
#endif /* DCT_CUH_ */
