%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% read image and compute disparity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd C:\Users\User\Desktop\MCL\scores
load 'mcl_3d_mos.mat';
dmos = mcl_3d_mos_score_nr';
dmos = 4*(-(min(dmos))+dmos);

for i=1:684
    
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


l=cell(1,684);
r=cell(1,684);
data_cla=cell(1,684);
lfolder = 'C:\Users\User\Desktop\MCL\left';
rfolder = 'C:\Users\User\Desktop\MCL\right';
for i=1:684
    
    k = int2str(i); 
    ln = [lfolder,'\1 (',k,').bmp'];
    rn = [rfolder,'\1 (',k,').bmp'];
    a=rgb2gray(imread(ln));
    b=rgb2gray(imread(rn));
    y=a-b;
    l{1,i} = a;
    r{1,i} = b;
    data_cla{1,i} = y;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% normalization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nor_l = cell(1,684);
Nor_r = cell(1,684);
Nor_d = cell(1,684);

for k=1:684

I1 = (double(l{1,k}));
I2 = (double(r{1,k}));
I3 = (double(data_cla{1,k}));

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% seg

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


data_l1 = cell(1,192*76);    %% 1-76
data_l2 = cell(1,510*152);  %% 77-228
data_l3 = cell(1,192*152);   %% 229-380
data_l4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(Nor_l{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_l1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(Nor_l{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_l2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(Nor_l{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_l3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(Nor_l{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_l4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end




data_r1 = cell(1,192*76);    %% 1-76
data_r2 = cell(1,510*152);  %% 77-228
data_r3 = cell(1,192*152);   %% 229-380
data_r4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(Nor_r{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_r1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(Nor_r{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_r2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(Nor_r{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_r3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(Nor_r{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_r4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end




data_d1 = cell(1,192*76);    %% 1-76
data_d2 = cell(1,510*152);  %% 77-228
data_d3 = cell(1,192*152);   %% 229-380
data_d4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(Nor_d{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_d1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(Nor_d{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_d2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(Nor_d{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_d3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(Nor_d{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_d4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end





data_cla = cell(1,276336*3);

for i=1:size(data_l1,2)
    
    data_cla{1,i} = data_l1{1,i};
    data_cla{1,i+276336} = data_r1{1,i};
    data_cla{1,i+276336*2} = data_d1{1,i};
    
end

for i=1:size(data_l2,2)
    
    data_cla{1,i+size(data_l1,2)} = data_l2{1,i};
    data_cla{1,i+size(data_r1,2)+276336} = data_r2{1,i};
    data_cla{1,i+size(data_d1,2)+276336*2} = data_d2{1,i};
    
end

for i=1:size(data_l3,2)
    
    data_cla{1,i+size(data_l1,2)+size(data_l2,2)} = data_l3{1,i};
    data_cla{1,i+size(data_l1,2)+size(data_r2,2)+276336} = data_r3{1,i};
    data_cla{1,i+size(data_l1,2)+size(data_d2,2)+276336*2} = data_d3{1,i};
    
end

for i=1:size(data_l4,2)
    
    data_cla{1,i+size(data_l1,2)+size(data_l2,2)+size(data_l3,2)} = data_l4{1,i};
    data_cla{1,i+size(data_l1,2)+size(data_l2,2)+size(data_r3,2)+276336} = data_r4{1,i};
    data_cla{1,i+size(data_l1,2)+size(data_l2,2)+size(data_d3,2)+276336*2} = data_d4{1,i};
    
end



s1 = ones(1,276336);
for i=1:8
    for j=1:192
        
        s1(1,((i-1)*10+4)*192+j) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+76*192) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+76*192+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s1(1,((i-1)*10+4)*192+j+76*192+76*510+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s1(1,((i-1)*10+4)*192+j+76*192+76*510+76*510+76*192) = 2;
        
    end
end

for i=1:7
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+76*192+76*510+76*510+76*192+76*192) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+76*192+76*150+76*510+76*192+76*192+7*510) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s1(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)+76*510) = 2;
        
    end
end

for i=1:76
    
    l(1,(i-1)*192+1:i*192) = group(1,i);
    labels_reg(1,(i-1)*192+1:i*192) = dmos(1,i);
end

for i=1:152
    
    l(1,(i-1)*510+1+76*192:i*510+76*192) = group(1,i+76*192);
    labels_reg(1,(i-1)*510+1+76*192:i*510+76*192) = dmos(1,i+76*192);
end

for i=1:152
    
    l(1,(i-1)*192+1+76*192+152*510:i*192+76*192+152*510) = group(1,i+76*192+152*510);
    labels_reg(1,(i-1)*192+1+76*192+152*510:i*192+76*192+152*510) = dmos(1,i+76*192+152*510);
end

for i=1:304
    
    l(1,(i-1)*510+1+76*192+152*510+152*192:i*510+76*192+152*510+152*192) = group(1,i+76*192+152*510+152*192);
    labels_reg(1,(i-1)*510+1+76*192+152*510+152*192:i*510+76*192+152*510+152*192) = dmos(1,i+76*192+152*510+152*192);
end

images = struct('data',cat(4,data_cla{:}),'labels',[l l l],'set',[s1 s1 s1]);
save im1.mat images -v7.3




s2 = ones(1,276336);

for i=1:8
    for j=1:192
        
        s2(1,((i-1)*10+4)*192+j) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+76*192) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+76*192+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s2(1,((i-1)*10+4)*192+j+76*192+76*510+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s2(1,((i-1)*10+4)*192+j+76*192+76*510+76*510+76*192) = 2;
        
    end
end

for i=1:7
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+76*192+76*510+76*510+76*192+76*192) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+76*192+76*150+76*510+76*192+76*192+7*510) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s2(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)+76*510) = 2;
        
    end
end

images = struct('data',cat(4,data_cla{:}),'labels',[l l l],'set',[s2 s2 s2]);
save im2.mat images -v7.3


s3 = ones(1,276336);

for i=1:8
    for j=1:192
        
        s3(1,((i-1)*10+4)*192+j) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+76*192) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+76*192+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s3(1,((i-1)*10+4)*192+j+76*192+76*510+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s3(1,((i-1)*10+4)*192+j+76*192+76*510+76*510+76*192) = 2;
        
    end
end

for i=1:7
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+76*192+76*510+76*510+76*192+76*192) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+76*192+76*150+76*510+76*192+76*192+7*510) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s3(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)+76*510) = 2;
        
    end
end

images = struct('data',cat(4,data_cla{:}),'labels',[l l l],'set',[s3 s3 s3]);
save im3.mat images -v7.3

s4 = ones(1,276336);

for i=1:8
    for j=1:192
        
        s4(1,((i-1)*10+4)*192+j) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+76*192) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+76*192+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s4(1,((i-1)*10+4)*192+j+76*192+76*510+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s4(1,((i-1)*10+4)*192+j+76*192+76*510+76*510+76*192) = 2;
        
    end
end

for i=1:7
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+76*192+76*510+76*510+76*192+76*192) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+76*192+76*150+76*510+76*192+76*192+7*510) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s4(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)+76*510) = 2;
        
    end
end

images = struct('data',cat(4,data_cla{:}),'labels',[l l l],'set',[s4 s4 s4]);
save im4.mat images -v7.3



s5 = ones(1,276336);

for i=1:8
    for j=1:192
        
        s5(1,((i-1)*10+4)*192+j) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+76*192) = 2;
        
    end
end
for i=1:8
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+76*192+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s5(1,((i-1)*10+4)*192+j+76*192+76*510+76*510) = 2;
        
    end
end
for i=1:8
    for j=1:192
        
        s5(1,((i-1)*10+4)*192+j+76*192+76*510+76*510+76*192) = 2;
        
    end
end

for i=1:7
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+76*192+76*510+76*510+76*192+76*192) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+76*192+76*150+76*510+76*192+76*192+7*510) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)) = 2;
        
    end
end
for i=1:7
    for j=1:510
        
        s5(1,((i-1)*10+4)*510+j+(76*192)+(76*510)+(76*510)+(76*192)+(76*192)+(7*510)+(76*510)+76*510) = 2;
        
    end
end

images = struct('data',cat(4,data_cla{:}),'labels',[l l l],'set',[s5 s5 s5]);
save im5.mat images -v7.3






%%%%%%%%%%%%%%%%%%%%%%%%%%

% data for regressor

%%%%%%%%%%%%%%%%%%%%%%%%%%%
im = cell(1,684);
odis = cell(1,684);

o1 = cell(1,684);
o2 = cell(1,684);
o3 = cell(1,684);

for p=1:76
        
    o = otsus(Nor_d{1,p},3);
    im1 = 255*ones(768,1024);
    im2 = zeros(768,1024);
    im3 = zeros(768,1024);
    
    for i=1:768
        for j=1:1024
            
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

for p=1:152
        
    o = otsus(Nor_d{1,p+76},3);
    im1 = 255*ones(1088,1920);
    im2 = zeros(1088,1920);
    im3 = zeros(1088,1920);
    
    for i=1:1088
        for j=1:1920
            
            if o(i,j) ==1
                im1(i,j) = Nor_d{1,p+76}(i,j);
                
            elseif o(i,j) ==2
                im2(i,j) = Nor_d{1,p+76}(i,j);
            else
                im3(i,j) = Nor_d{1,p+76}(i,j);
            end
            
        end
    end
    
    o1{1,p+76} = im1;
    o2{1,p+76} = im2;
    o3{1,p+76} = im3;
    
end

for p=1:152
        
    o = otsus(Nor_d{1,p+76+152},3);
    im1 = 255*ones(768,1024);
    im2 = zeros(768,1024);
    im3 = zeros(768,1024);
    
    for i=1:768
        for j=1:1024
            
            if o(i,j) ==1
                im1(i,j) = Nor_d{1,p+76+152}(i,j);
                
            elseif o(i,j) ==2
                im2(i,j) = Nor_d{1,p+76+152}(i,j);
            else
                im3(i,j) = Nor_d{1,p+76+152}(i,j);
            end
            
        end
    end
    
    o1{1,p+76+152} = im1;
    o2{1,p+76+152} = im2;
    o3{1,p+76+152} = im3;
    
end

for p=1:304
        
    o = otsus(Nor_d{1,p+76+304},3);
    im1 = 255*ones(1088,1920);
    im2 = zeros(1088,1920);
    im3 = zeros(1088,1920);
    
    for i=1:1088
        for j=1:1920
            
            if o(i,j) ==1
                im1(i,j) = Nor_d{1,p+76+304}(i,j);
                
            elseif o(i,j) ==2
                im2(i,j) = Nor_d{1,p+76+304}(i,j);
            else
                im3(i,j) = Nor_d{1,p+76+304}(i,j);
            end
            
        end
    end
    
    o1{1,p+76+304} = im1;
    o2{1,p+76+304} = im2;
    o3{1,p+76+304} = im3;
    
end


data_o1d1 = cell(1,192*76);    %% 1-76
data_o1d2 = cell(1,510*152);  %% 77-228
data_o1d3 = cell(1,192*152);   %% 229-380
data_o1d4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(o1{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o1d1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(o1{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o1d2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(o1{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o1d3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(o1{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o1d4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


data_o2d1 = cell(1,192*76);    %% 1-76
data_o2d2 = cell(1,510*152);  %% 77-228
data_o2d3 = cell(1,192*152);   %% 229-380
data_o2d4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(o2{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o2d1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(o2{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o2d2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(o2{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o2d3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(o2{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o2d4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


data_o3d1 = cell(1,192*76);    %% 1-76
data_o3d2 = cell(1,510*152);  %% 77-228
data_o3d3 = cell(1,192*152);   %% 229-380
data_o3d4 = cell(1,510*304);  %% 381-684

for p=1:76
    
  
Im = (single(o3{1,p}));


L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o3d1{1,(p-1)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76:152+76
    
  
Im = (single(o3{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o3d2{1,(p-1-76)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



for p=1+76+152:152+76+152
    
  
Im = (single(o3{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o3d3{1,(p-1-76-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end


for p=1+76+152+152:304+76+152+152
    
  
Im = (single(o3{1,p}));
L = size(Im);

max_r = L(1)/64;
max_c = L(2)/64;
seg = cell(max_r,max_c);

for row = 1:max_r      
    for col = 1:max_c
        seg{row,col}= Im((row-1)*64+1:row*64,(col-1)*64+1:col*64);   
    end
end 
    
seg = reshape(seg,1,L(1)*L(2)/64/64);


for i=1:L(1)*L(2)/64/64
    data_o3d4{1,(p-1-76-152-152)*L(1)*L(2)/64/64+i} = seg{1,i};
end


end



data_reg = cell(1,276336*5);

for i=1:size(data_l1,2)
    
    data_reg{1,i} = data_l1{1,i};
    data_reg{1,i+276336} = data_r1{1,i};
    data_reg{1,i+276336*2} = data_o1d1{1,i};
    data_reg{1,i+276336*3} = data_o2d1{1,i};
    data_reg{1,i+276336*4} = data_o3d1{1,i};
    
end

for i=1:size(data_l2,2)
    
    data_reg{1,i+size(data_l1,2)} = data_l2{1,i};
    data_reg{1,i+size(data_l1,2)+276336} = data_r2{1,i};
    data_reg{1,i+size(data_l1,2)+276336*2} = data_o1d2{1,i};
    data_reg{1,i+size(data_l1,2)+276336*3} = data_o2d2{1,i};
    data_reg{1,i+size(data_l1,2)+276336*4} = data_o3d2{1,i};
    
end

for i=1:size(data_l3,2)
    
    data_reg{1,i+size(data_l1+size(data_l2,2),2)} = data_l3{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+276336} = data_r3{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+276336*2} = data_o1d3{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+276336*3} = data_o2d3{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+276336*4} = data_o3d3{1,i};
    
end

for i=1:size(data_l4,2)
    
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+size(data_l3,2)} = data_l4{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+size(data_l3,2)+276336} = data_r4{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+size(data_l3,2)+276336*2} = data_o1d4{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+size(data_l3,2)+276336*3} = data_o2d4{1,i};
    data_reg{1,i+size(data_l1+size(data_l2,2),2)+size(data_l3,2)+276336*4} = data_o3d4{1,i};
        
end

save data_reg_mcl.mat data_reg -v7.3