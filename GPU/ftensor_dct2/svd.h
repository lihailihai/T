/*
 * svd.h
 *
 *  Created on: Jan 14, 2018
 *      Author: haili
 */

#ifndef SVD_H_
#define SVD_H_
#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include<assert.h>
#include<cuda_runtime.h>
#include"cusolverDn.h"
void svd(int,int,cuComplex*,cuComplex*,cuComplex*,float*);
void svd_getv(int m,int n,cuComplex*,cuComplex*,cuComplex*,float*);
#endif /* SVD_H_ */
