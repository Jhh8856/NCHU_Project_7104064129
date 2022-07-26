load net-epoch-70


res = [];
dzdy = [] ;
h=cell(1,780);

for i=1:780

nett = vl_simplenn_move(net, 'gpu');    
I = gpuArray(images.data(:,:,1,(i-1)*12+1:i*12));
res = vl_simplenn(nett, I ,res,dzdy,'mode','test') ;

h{i} = gather(squeeze(res(end).x));
res=[];
gpuDevice(1);
end