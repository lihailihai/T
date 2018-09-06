#!/usr/bin/env python3
#_*_coding:utf-8_*_
import numpy as np
import os
import sys
import fft_svd
import dct_svd
import dwt_svd
def main():
#    t=np.random.rand(2,3,4,4)
    #------------tensor------------------------------------------------
    t=np.array([[[[1,2,3],[4,5,6]],[[3,4,5],[2,3,4]]],[[[7,8,9],[10,11,12]],[[8,9,0],[5,6,7]]]])
    print('==============tensor========================\n',t)
    #------------------------------------------------------------------


    #-----------test fft_svd-------------------------------------------
    u,s,v=fft_svd.svd(t)
    print('===================fft_svd======================\n')
    print('outputs have fft applied along  dimensions of third and fourth.\n ')
    print('-------------------fft_svd_U------------------\n',u)
    print('-------------------fft_svd_S------------------\n',s)
    print('-------------------fft_svd_V------------------\n',v)
    print('================================================\n')
    #------------------------------------------------------------------



    #-----------test dct_svd-------------------------------------------
    u,s,v=dct_svd.svd(t)
    print('===================dct_svd======================\n')
    print('outputs have dct applied along  dimensions of third and fourth.\n ')
    print('-------------------dct_svd_U------------------\n',u)
    print('-------------------dct_svd_S------------------\n',s)
    print('-------------------dct_svd_V------------------\n',v)
    print('================================================\n')
    

    #-----------test dwt_svd-------------------------------------------
    u,s,v=dwt_svd.svd(t)
    print('===================dwt_svd======================\n')
    print('outputs have dwt applied along  dimensions of third and fourth.\n ')
    print('-------------------dwt_svd_U------------------\n',u)
    print('-------------------dwt_svd_S------------------\n',s)
    print('-------------------dwt_svd_V------------------\n',v)
    print('================================================\n')
if __name__=='__main__':
   main()
