%==================load========================
close all
clear all
clc
a=load('/home/haili/Documents/python3/cpu_based_dct2/com_result9_10.txt');
b=load('/home/haili/Documents/MATLAB/gpu_based_dct2/compress_result9_10.txt');
load('/home/haili/Documents/MATLAB/video_process/rgbdata.mat')
load('/home/haili/Documents/MATLAB/video_process/data1.mat')
%==============================================
%%
[a1,a2,a3,a4]=size(rgbdata);
result1=zeros(a1,a2,a3,a4);
result2=zeros(a1,a2,a3,a4);
for i = 1:a4
    for j = 1:a3
        for k = 1:a1
            for l = 1:a2
          result1(k,l,j,i)=uint8(a(l+(k-1)*a2+(j-1)*a1*a2+(i-1)*a1*a2*a3));
          result2(k,l,j,i)=uint8(b(l+(k-1)*a2+(j-1)*a1*a2+(i-1)*a1*a2*a3));
            end
        end
    end
end
%=================================================
 %%
%=================plot============================
figure;
for i =1:(endl-start) 
subplot(221);imshow(rgbdata(:,:,:,i));axis off;
title('orign');
subplot(222);imshow(result1(:,:,:,i)/max(result1(:)));axis off;
title('compress python');
subplot(224);imshow(result2(:,:,:,i)/max(result2(:)));axis off;
title('compress cuda');
pause(.1);
end
%%
%===================save data======================
save('T1.mat' ,'result1');