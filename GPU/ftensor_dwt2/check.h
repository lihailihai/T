/*
 * check.h
 *
 *  Created on: Jan 11, 2018
 *      Author: haili
 */

#ifndef CHECK_H_
#define CHECK_H_
#include"common.h"
extern void checkcusparse(cusparseStatus_t err);
extern void checkcusolverDn(cusolverStatus_t err);
extern void checkruntime(cudaError_t err);
extern void checkkernel();
extern void checkfft(cufftResult err);
#endif /* CHECK_H_ */
