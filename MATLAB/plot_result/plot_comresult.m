%==================load compress data=============
close all
clear all
clc
load('/home/haili/Documents/MATLAB/gpu_based_dct2/compress_result.txt');
load('/home/haili/Documents/MATLAB/video_process/rgbdata.mat')
load('/home/haili/Documents/MATLAB/video_process/data1.mat')
%==================================================
%%
[a1,a2,a3,a4]=size(rgbdata);
result=zeros(a1,a2,a3,a4);
for i = 1:a4
    for j = 1:a3
        for k = 1:a1
            for l = 1:a2
          result(k,l,j,i)=uint8(compress_result(l+(k-1)*a2+(j-1)*a1*a2+(i-1)*a1*a2*a3));
            end
        end
    end
end
%=================================================
%%
%=================plot============================
figure;
for i =1:(endl-start) 
subplot(121);imshow(rgbdata(:,:,:,i));axis off;
title('orign');
subplot(122);imshow(result(:,:,:,i)/max(result(:)));axis off;
title('compress');
pause(.1);
end
%%
%===================save data======================
save('T.mat' ,'result');
