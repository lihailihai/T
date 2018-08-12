#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
__global__ void d_batch_transpose(float*,float*,const int,const int,const int);
#endif
