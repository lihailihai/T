/*
 * multi_mm.h
 *
 *  Created on: Jan 16, 2018
 *      Author: haili
 */

#ifndef MULTI_MM_H_
#define MULTI_MM_H_
#include<stdio.h>
#include<assert.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include"tensor.h"
void multi_mm(
		int* A_row_array,
		int* A_col_array,
		cuComplex* A_data_array,
		int* B_row_array,
		int* B_col_array,
		cuComplex* B_data_array,
		int m,
		int n,
		int min,
		int k,
		int l,
		cusparseOperation_t transA,
		cusparseOperation_t transB
		);
#endif /* MULTI_MM_H_ */
