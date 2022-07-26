load net-epoch-147
a = find(images.set==2);
im=cell(size(a));

for i=1:size(a,2)
    
    im{1,i} = images.data(:,:,1,a(1,i)); 
    for j=1
        l(j,i) = images.labels(j,a(1,i));
    end
    
end


I = (cat(4,im{:}));
res = [];
dzdy = [] ;
h=cell(1,10);

for i=1:10
    

res = vl_simplenn(net, I(:,:,1,(i-1)*1296+1:i*1296) ,res,dzdy,'mode','test') ;

h{i} = squeeze(res(end).x);
res=[];

end