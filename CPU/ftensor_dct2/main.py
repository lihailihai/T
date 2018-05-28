#!/usr/bin/env python3
#_*_coding:utf-8_*_
'fourth tensor SVD'
__author__='hai.li'
import os
import numpy as np
import sys
from scipy.fftpack import dct, idct
sys.path.append("/home/haili/tensor")
import tn_scalar #ac
def main():
    data=[]
    f=open('/home/haili/Documents/MATLAB/kobe.txt','r')
    for line in f.readlines():
        data.append(list(map(float,line.split())))
    f.close
#    print(data)
    #create tensor of t

    m,n,k,l=eval(input('enter the m n k l:'))
    print(m,n,k,l)
#   t=np.random.random_sample((m*n*k*l))*100
    te=np.array(data).flatten()
    print(te)
    print(type(te))

    # tensor to tensor-scalar
    ts=tn_scalar.tensor_scalar(te,m,n,k,l)
 #   print(ts)
    temp=np.empty([m*n*l,k])
    temp2=np.empty([m*n*k,l])
    # a set of tensor-scalar do  dct with k
    for i in range(m*n*l):
        t_array=ts[i*k:k*(i+1)]
        temp[i][:]=dct(t_array,type=2,norm='ortho')
       # if i==0:
       #     tf=dct(t_array,type=2,norm='ortho')
        #else:
        #    tf=np.vstack((tf,dct(t_array,type=2,norm='ortho')))
#    print(tf,type(tf))
    tf=temp.flatten()
#    print(tf,type(tf))
    ##tensor scalar tranpose
    tf_transpose=tn_scalar.tensor_scalar_transpose(m,n,k,l,tf)
#    print(tf_transpose,type(tf_transpose))
    ##take dct with l
    for i in range(m*n*k):
        t_array1=tf_transpose[i*l:l*(i+1)]
        temp2[i][:]=dct(t_array1,type=2,norm='ortho')
       # if i == 0:
        #    t_array2=dct(t_array1,type=2,norm='ortho')
       # else:
        #    t_array2=np.vstack((t_array2,dct(t_array1,type=2,norm='ortho')))
    tf=temp2.flatten()
    tf4=tn_scalar.tensor_scalar_transpose(m,n,l,k,tf)
    # tensor-scalar to tensor
    tf1=tn_scalar.scalar_tensor(tf4,m,n,k,l)
#    print('tf1',tf1)

    # a set of M do SVD
    for i in range(k*l):
        u,s,v=np.linalg.svd(tf1[m*n*i:m*n*(i+1)].reshape((m,n)))
        if i==0:
            U,S,V=u,s,v
        else:
            U,V,S=np.vstack((U,u)),np.vstack((V,v)),np.vstack((S,s))
    print('U…………………………\n',U) 
    print('S…………………………\n',S) 
    print('V…………………………\n',V)
 #   r=0
    print('size \n',S.size)
     
    r=(np.size(S)*9)//10
    print('r \n',r)
    S1=S.flatten()
    tn_scalar.compress(S1,r)
    print('compress S\n',S1)

    #U*S*V
    if m>n:
        t=n
   #     S1=S.flatten()
        V1=V.flatten()
        U1=U[:,0:t].flatten()
        for i in range(k*l):
            d=np.dot(U1[m*t*i:m*t*(i+1)].reshape(m,t),np.diag(S1[t*i:t*(i+1)]))
            re=np.dot(d,V1[n*n*i:n*n*(i+1)].reshape(n,n))
            if i==0:
                result=re
            else:
                result=np.vstack((result,re))
    else:
        t=m
    #    S1=S.flatten()
        U1=U.flatten()
        V1=V.flatten()
        for i in range(k*l):
            d=np.dot(U1[m*m*i:m*m*(i+1)].reshape(m,m),np.diag(S1[t*i:t*(i+1)]))
            re=np.dot(d,V1[n*n*i:(n*n*i+t*n)].reshape(t,n))
            if i==0:
                result=re
            else:
                result=np.vstack((result,re))
#    result1=[int(result.flatten()[i]-tf1[i]) for i in range(n*m*k*l)]
#    print('result\n',result)
    res=result.flatten()
#    res2=res.tolist()
     # tensor to tensor-scalar
    ts=tn_scalar.tensor_scalar(res,m,n,k,l)
#    print(ts)
    res=tn_scalar.tensor_scalar_transpose(m,n,k,l,ts)
    # a set of tensor-scalar do  dct
    for i in range(m*n*k):
        t_array=res[i*l:l*(i+1)]
        temp2[i][:]=idct(t_array,type=2,norm='ortho')
     #   if i==0:
      #      tf=idct(t_array,type=2,norm='ortho')
       # else:
        #    tf=np.vstack((tf,idct(t_array,type=2,norm='ortho')))
#    print(tf,type(tf))
    ## tensor scalar transpose
    tf=temp2.flatten()
    ts=tn_scalar.tensor_scalar_transpose(m,n,l,k,tf)
    for i in range(m*n*l):
        t_array=ts[i*k:k*(i+1)]
        temp[i][:]=idct(t_array,type=2,norm='ortho')
       # if i== 0:
        #    tf=idct(t_array,type=2,norm='ortho')
        #else:
         #   tf=np.vstack((tf,idct(t_array,type=2,norm='ortho')))
    tf=temp.flatten()
#    print(tf,type(tf))

    # tensor-scalar to tensor
    tf2=tn_scalar.scalar_tensor(tf,m,n,k,l)
#    print(tf2)
    tf3=tf2.real
#    np.savetxt('/home/haili/Documents/python3/com_result.txt',tf2)
    with open('/home/haili/Documents/python3/cpu_based_dct2/com_result9_10.txt','w') as ft:
        for i in range(m*n*k*l):
            ft.write(str(tf3[i])+'\n')
#    result1=tf2.real
#    print(te)
#    print('result1\n',result1)
if __name__=='__main__':
    main()
