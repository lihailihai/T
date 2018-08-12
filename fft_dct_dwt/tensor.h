/*
 * tensor.h
 *
 *  Created on: Jan 11, 2018
 *      Author: haili
 */

#ifndef TENSOR_H_
#define TENSOR_H_
#include<stdio.h>
#include<assert.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
float* t_scalar_transpose(int k,int l,int batch,const float* t,float* T);
#endif /* TENSOR_H_ */
