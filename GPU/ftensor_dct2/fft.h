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
void fft(int m,int n,int k,int l,cufftComplex *t,cufftComplex* ft);
void ifft(int m,int n,int k,int l,cufftComplex *t,cufftComplex* ft);
void fft_batch(int,int,int,int,cuComplex*,cuComplex*);
void ifft_batch(int,int,int,int,cuComplex*,cuComplex*);
#endif /* FFT_H_*/
