

for i=1:size(a1,2)
    d1(:,:,1,i) = images.data(:,:,1,a1(1,i));
    l1(1,i) = l(1,a1(1,i));
end

for i=1:size(a2,2)
    d2(:,:,1,i) = images.data(:,:,1,a2(1,i));
    l2(1,i) = l(1,a2(1,i));
end

for i=1:size(a3,2)
    d3(:,:,1,i) = images.data(:,:,1,a3(1,i));
    l3(1,i) = l(1,a3(1,i));
end

for i=1:size(a4,2)
    d4(:,:,1,i) = images.data(:,:,1,a4(1,i));
    l4(1,i) = l(1,a4(1,i));
end

s1 = ones(1,size(a1,2));
s1(1,size(a1,2)-10+1:size(a1,2))=2;

s2 = ones(1,size(a2,2));
s2(1,size(a2,2)-10+1:size(a2,2))=2;

s3 = ones(1,size(a3,2));
s3(1,size(a3,2)-10+1:size(a3,2))=2;

s4 = ones(1,size(a4,2));
s4(1,size(a4,2)-10+1:size(a4,2))=2;

images = struct('data',d1,'labels',l1,'set',s1);
save img1.mat images

images = struct('data',d2,'labels',l2,'set',s2);
save img2.mat images

images = struct('data',d3,'labels',l3,'set',s3);
save img3.mat images

images = struct('data',d4,'labels',l4,'set',s4);
save img4.mat images