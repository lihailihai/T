/*
 * fft.h
 *
 *  Created on: Jan 12, 2018
 *      Author: haili
 */

#ifndef FFT_H_
#define FFT_H_
#include<cufft.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<cuda_runtime.h>
void fft2_batch(int,int,cuComplex*,cuComplex*,int);
void ifft2_batch(int,int,cuComplex*,cuComplex*,int);
#endif /* FFT_H_*/

