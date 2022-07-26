load net-epoch-75


res = [];
dzdy = [] ;
h=cell(1,1800);

for i=1:1800

nett = vl_simplenn_move(net, 'gpu');    
I = gpuArray(images.data(:,:,1,(i-1)*12+1:i*12));
res = vl_simplenn(nett, I ,res,dzdy,'mode','test') ;

h{i} = gather(squeeze(res(end).x));
res=[];
gpuDevice(1);
end