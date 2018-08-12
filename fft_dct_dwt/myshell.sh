#!/bin/bash
nvidia-smi
cd /home/haili/Documents/fft_dct_dwt/debug
touch result.txt
m=200
n=200
k=32
i=0
j=0
p=0
while [ $p -lt 10 ]; do
while [ $j -lt 1 ]; do
while [ $i -lt 10 ]; do
#   echo "$m $n $k"
   ./result $m $n $k  >> result.txt
   m=`expr $m + 200` n=`expr $n + 200`
   i=`expr $i + 1`
done
   k=`expr $k + $k`
   m=200
   n=200
   i=0
   j=`expr $j + 1`
done
   j=0
   k=32
   p=`expr $p + 1`
done
exit 0
