#!/usr/bin/env python3
#_*_coding:utf-8_*_
'fourth tensor SVD'
__author__='hai.li'
import os
import numpy as np
import sys
import pywt
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
    k1=(k+7)//2
    l1=(l+7)//2
    temp=np.empty([m*n,k1*l1*4])
    temp1=np.empty([m*n,k*l])
    # a set of tensor-scalar do  dwt2 
    for i in range(m*n):
        t_array=ts[i*k*l:k*l*(i+1)].reshape(l,k)
        coeffs=pywt.dwt2(t_array,'db4')
        temp[i][:]=np.concatenate((np.concatenate((coeffs[0],coeffs[1][0]),axis=1),np.concatenate((coeffs[1][1],coeffs[1][2]),axis=1))).flatten()
#    print(tf,type(tf))
    tf=temp.flatten()
#    print(tf,type(tf))
    # tensor-scalar to tensor
    tf1=tn_scalar.scalar_tensor(tf,m,n,k1*2,l1*2)
#    print('tf1',tf1)

    # a set of M do SVD
    for i in range(k1*l1*4):
        u,s,v=np.linalg.svd(tf1[m*n*i:m*n*(i+1)].reshape((m,n)))
        if i==0:
            U,S,V=u,s,v
        else:
            U,V,S=np.vstack((U,u)),np.vstack((V,v)),np.vstack((S,s))
    print('U…………………………\n',U) 
    print('S…………………………\n',S) 
    print('V…………………………\n',V)
#    r=0
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
        for i in range(k1*l1*4):
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
        for i in range(k1*l1*4):
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
    ts=tn_scalar.tensor_scalar(res,m,n,k1*2,l1*2)
#    print(ts)
    # a set of tensor-scalar do  idwt2
    for i in range(m*n):
        t_array=ts[i*k1*l1*4:k1*l1*4*(i+1)].reshape(l1*2,k1*2)
        icoeffs=tuple((t_array[0:l1,0:k1],tuple((t_array[0:l1,k1:2*k1],t_array[l1:2*l1,0:k1],t_array[l1:2*l1,k1:2*k1]))))
#        print('test\n',pywt.idwt2(icoeffs,'db4'))
        temp1[i][:]=pywt.idwt2(icoeffs,'db4')[0:l,0:k].flatten()
    tf=temp1.flatten()
#    print(tf,type(tf))

    # tensor-scalar to tensor
    tf2=tn_scalar.scalar_tensor(tf,m,n,k,l)
#    print(tf2)
    tf3=tf2.real
#    np.savetxt('/home/haili/Documents/python3/com_result.txt',tf2)
    with open('/home/haili/Documents/python3/cpu_based_dwt2/com_result9_10.txt','w') as ft:
        for i in range(m*n*k*l):
            ft.write(str(tf3[i])+'\n')
#    result1=tf2.real
#    print(te)
#    print('result1\n',result1)
if __name__=='__main__':
    main()
