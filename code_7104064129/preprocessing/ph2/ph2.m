
cd C:\Users\User\Desktop\LIVE3DIQAPhase2
load 3DDmosRelease

dmos = Dmos';
dmos = -(min(dmos))+dmos;
for i=1:360
    
    if dmos(1,i) <12
        group(1,i) = 1;
    elseif dmos(1,i)>=12 && dmos(1,i)<24
        group(1,i) = 2;
    elseif dmos(1,i)>=24 && dmos(1,i)<36
        group(1,i) = 3;
    elseif dmos(1,i)>=36 
        group(1,i) = 4;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% read image and compute disparity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l = cell(1,360);
r = cell(1,360);
d = cell(1,360);



for  j = 1:360
    
     m = int2str(j);
    imname = ['C:\Users\User\Desktop\LIVE3DIQAPhase2\stimuli',StiFilename];  
     
    I = single(imread(imname));
    
    l{1,j} = I(:,1:640,:);
    r{1,j} = I(:,641:1280,:);
    d{1,j} = I(:,1:640,:) - I(:,641:1280,:);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% normalization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Nor_l = cell(1,360);
Nor_r = cell(1,360);
Nor_d = cell(1,360);

for k=1:360

I1 = (double(l{1,k}));
I2 = (double(r{1,k}));
I3 = (double(d{1,k}));

Im1 = padarray(I1,[3 3],'symmetric');
Im2 = padarray(I1,[3 3],'symmetric');
Im3 = padarray(I1,[3 3],'symmetric');

[h w] = size(Im1);


for i=4:h-3
             
    for j=4:w-3
        u1(i,j)=(sum(sum(Im1(i-3:i+3,j-3:j+3))))/49;
        u2(i,j)=(sum(sum(Im2(i-3:i+3,j-3:j+3))))/49;
        u3(i,j)=(sum(sum(Im3(i-3:i+3,j-3:j+3))))/49;
    end
    
end


for i=4:h-3
    
     for j=4:w-3
         sigma1(i,j) = (sqrt(sum(sum((Im1(i-3:i+3,j-3:j+3)-u1(i,j)).^2))))/49;
         sigma2(i,j) = (sqrt(sum(sum((Im2(i-3:i+3,j-3:j+3)-u3(i,j)).^2))))/49; 
         sigma3(i,j) = (sqrt(sum(sum((Im3(i-3:i+3,j-3:j+3)-u3(i,j)).^2))))/49; 
     end
     
end

p=1:3;
u1(p,:)=[];
u1(:,p)=[];
u2(p,:)=[];
u2(:,p)=[];
u3(p,:)=[];
u3(:,p)=[];

sigma1(p,:)=[];
sigma1(:,p)=[];
sigma2(p,:)=[];
sigma2(:,p)=[];
sigma3(p,:)=[];
sigma3(:,p)=[];


Nor1 = (I1-(u1))./(sigma1);
Nor2 = (I2-(u2))./(sigma2);
Nor3 = (I3-(u3))./(sigma3);

Nor_l{1,k} = Nor1;
Nor_r{1,k} = Nor2;
Nor_d{1,k} = Nor3;

end




seg_clas = cell(1,21600*3);

for p=1:360
    
  
Im = (single(Nor_l{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
seg_clas{(p-1)*60+i} = seg{1,i};
end


end


for p=1:360
    
  
Im = (single(Nor_r{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
seg_clas{(p-1)*60+i+21600} = seg{1,i};
end


end


for p=1:360
    
  
Im = (single(Nor_d{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
seg_clas{(p-1)*60+i+21600*2} = seg{1,i};
end
end

for i=1:360
    
    l(1,(i-1)*60+1:i*60) = group(1,i);
    
end

s1 = ones(1,21600*3);
s2 = ones(1,21600*3);
s3 = ones(1,21600*3);
s4 = ones(1,21600*3);
s5 = ones(1,21600*3);
for i=1:72
   
    s1(1,((i-1)*5)*60+1:((i-1)*5)*60+60)=2;
    s2(1,((i-1)*5)*60+61:((i-1)*5)*60+120)=2;
    s3(1,((i-1)*5)*60+121:((i-1)*5)*60+180)=2;
    s4(1,((i-1)*5)*60+181:((i-1)*5)*60+240)=2;
    s5(1,((i-1)*5)*60+241:((i-1)*5)*60+300)=2;
    
end


images = struct('data',cat(4,seg_clas{:}),'labels',[l l l],'set',s1);
save im1.mat images -v7.3

images = struct('data',cat(4,seg_clas{:}),'labels',[l l l],'set',s2);
save im2.mat images -v7.3

images = struct('data',cat(4,seg_clas{:}),'labels',[l l l],'set',s3);
save im3.mat images -v7.3

images = struct('data',cat(4,seg_clas{:}),'labels',[l l l],'set',s4);
save im4.mat images -v7.3

images = struct('data',cat(4,seg_clas{:}),'labels',[l l l],'set',s5);
save im5.mat images -v7.3

%%%%%%%%%%%%%%%%%%%%%%%%%%

% data for regressor

%%%%%%%%%%%%%%%%%%%%%%%%%%%
im = cell(1,360);
odis = cell(1,360);

o1 = cell(1,360);
o2 = cell(1,360);
o3 = cell(1,360);

for p=1:360
        
    o = otsus(Nor_d{1,p},3);
    im1 = 255*ones(360,640);
    im2 = zeros(360,640);
    im3 = zeros(360,640);
    
    for i=1:360
        for j=1:640
            
            if o(i,j) ==1
                im1(i,j) = Nor_d{1,p}(i,j);
                
            elseif o(i,j) ==2
                im2(i,j) = Nor_d{1,p}(i,j);
            else
                im3(i,j) = Nor_d{1,p}(i,j);
            end
            
        end
    end
    
    o1{1,p} = im1;
    o2{1,p} = im2;
    o3{1,p} = im3;
    
end


data_reg = cell(1,21600*5);

for p=1:360
    
  
Im = (single(Nor_l{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
data_reg{(p-1)*60+i} = seg{1,i};
end
end


for p=1:360
    
  
Im = (single(Nor_r{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
data_reg{(p-1)*60+i+21600} = seg{1,i};
end
end


for p=1:360
    
  
Im = (single(o1{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
data_reg{(p-1)*60+i+21600*2} = seg{1,i};
end
end


for p=1:360
    
  
Im = (single(o2{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
data_reg{(p-1)*60+i+21600*3} = seg{1,i};
end
end


for p=1:360
    
  
Im = (single(o3{1,p}));
L = size(Im);
height=64;
width=64;
max_row = 6;
max_col = 10;
seg = cell(max_row,max_col);

for row = 1:5      
    for col = 1:10
        seg(row,col)= {Im((row-1)*height+1:row*height,(col-1)*width+1:col*width,:)};   
    end
end 
 for row = 6
     for col =1:10
         seg(row,col) = {Im(329:360,(col-1)*width+1:col*width,:)};
     end
 end
    
seg = reshape(seg,1,60);
for i=1:60
data_reg{(p-1)*60+i+21600*4} = seg{1,i};
end
end


save data_reg_ph2.mat data_reg -v7.3