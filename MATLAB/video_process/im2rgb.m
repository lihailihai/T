%========================================
%  image to data with rgb
%========================================

start=191;   %frame 100 : 200
endl=200;

%=================get rgb data===========
%%
for i = start:endl
    
   a = imread(strcat('/home/haili/Documents/MATLAB/image/',num2str(i),'.jpg'));
   
   if i== start
       rgbdata = a;
   else
       rgbdata=cat(4,rgbdata,a);
   end
end
save('rgbdata.mat','rgbdata');
save('data1.mat', 'start' ,'endl')
%=====================plot==============
%%
figure;
for i= 1:size(rgbdata,4)
    subplot(221);
    imshow(rgbdata(:,:,:,i));
    title('rgb');axis off;
    subplot(222);
    imshow(rgbdata(:,:,1,i));
    title('R');axis off;
    subplot(223);
    imshow(rgbdata(:,:,2,i));
    title('G');axis off;
    subplot(224);
    imshow(rgbdata(:,:,3,i));
    title('B');axis off;
    pause(.1);
end
%===============save data==================
%%
fid=fopen('/home/haili/Documents/MATLAB/kobe.txt','wt');

[b1,b2,b3,b4]=size(rgbdata);
for i=1:b4
    for j=1:b3
        for k=1:b1
            for l=1:b2
                fprintf(fid,'%f\n',rgbdata(k,l,j,i));
            end
        end
    end
end

fclose(fid);
                
%==========================================
    