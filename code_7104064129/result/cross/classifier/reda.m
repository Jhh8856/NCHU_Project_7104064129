t = size(data_reg,2)/5;
l = labels+reg;
s = images.set;

for i=1:size(a1,2)/3
    
    d1(:,:,1,i) = data_reg(:,:,1,a1(1,i));
    d1(:,:,1,i+size(a1,2)/3) = data_reg(:,:,1,a1(1,i)+t);
    d1(:,:,1,i+2*size(a1,2)/3) = data_reg(:,:,1,a1(1,i)+2*t);
    d1(:,:,1,i+3*size(a1,2)/3) = data_reg(:,:,1,a1(1,i)+3*t);
    d1(:,:,1,i+4*size(a1,2)/3) = data_reg(:,:,1,a1(1,i)+4*t);
    
    l1(1,i) = l(1,a1(1,i));
    s1(1,i) = s(1,a1(1,i));
end

for i=1:size(a2,2)/3
    
    d2(:,:,1,i) = data_reg(:,:,1,a2(1,i));
    d2(:,:,1,i+size(a2,2)/3) = data_reg(:,:,1,a2(1,i)+t);
    d2(:,:,1,i+2*size(a2,2)/3) = data_reg(:,:,1,a2(1,i)+2*t);
    d2(:,:,1,i+3*size(a2,2)/3) = data_reg(:,:,1,a2(1,i)+3*t);
    d2(:,:,1,i+4*size(a2,2)/3) = data_reg(:,:,1,a2(1,i)+4*t);
    
    l2(1,i) = l(1,a2(1,i));
    s2(1,i) = s(1,a2(1,i));
end

for i=1:size(a3,2)/3
    
    d3(:,:,1,i) = data_reg(:,:,1,a3(1,i));
    d3(:,:,1,i+size(a1,3)/3) = data_reg(:,:,1,a3(1,i)+t);
    d3(:,:,1,i+2*size(a3,2)/3) = data_reg(:,:,1,a3(1,i)+2*t);
    d3(:,:,1,i+3*size(a3,2)/3) = data_reg(:,:,1,a3(1,i)+3*t);
    d3(:,:,1,i+4*size(a3,2)/3) = data_reg(:,:,1,a3(1,i)+4*t);
    
    l3(1,i) = l(1,a3(1,i));
    s3(1,i) = s(1,a3(1,i));
end

for i=1:size(a4,2)/3
    
    d4(:,:,1,i) = data_reg(:,:,1,a1(1,i));
    d4(:,:,1,i+size(a4,2)/3) = data_reg(:,:,1,a4(1,i)+t);
    d4(:,:,1,i+2*size(a4,2)/3) = data_reg(:,:,1,a4(1,i)+2*t);
    d4(:,:,1,i+3*size(a4,2)/3) = data_reg(:,:,1,a4(1,i)+3*t);
    d4(:,:,1,i+4*size(a4,2)/3) = data_reg(:,:,1,a4(1,i)+4*t);
    
    l4(1,i) = l(1,a4(1,i));
    s4(1,i) = s(1,a4(1,i));
end




images = struct('data',d1,'labels',[l1 l1 l1 l1 l1],'set',[s1 s1 s1 s1 s1]);
save img1.mat images

images = struct('data',d2,'labels',[l2 l2 l2 l2 l2],'set',[s2 s2 s2 s2 s2]);
save img2.mat images

images = struct('data',d3,'labels',[l3 l3 l3 l3 l3],'set',[s3 s3 s3 s3 s3]);
save img3.mat images

images = struct('data',d4,'labels',[l4 l4 l4 l4 l4],'set',[s4 s4 s4 s4 s4]);
save img4.mat images



