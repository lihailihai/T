%=================load video===============
clear all
close all
clc 
kobe = VideoReader('/home/haili/Videos/kobe.mp4');

kobe_numberofframe = kobe.NumberOfFrame;

kobe_height = kobe.Height;
%================read frame================
%frame = read(kobe);  %   all  frame
%frame = read(kobe,[1,10]);        %  1 to 10  of frame
%frame = read(kobe,Inf);           %   last one of frame
%frame = read(kobe,[50,Inf]);

%===============imwrite====================

%imwrite(frame,strcat('~/Documents/MATLAB/name.jpg','jpg'));
Start=100;   %frame from $Satrt  to   $End 
End=200;

for k = Start:End
    frame = read(kobe,k);
    imshow(frame);
    imwrite(frame,strcat('/home/haili/Documents/MATLAB/image/',num2str(k),'.jpg'),'jpg');
end
