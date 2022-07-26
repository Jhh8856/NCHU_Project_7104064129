load net-epoch-60


res = [];
dzdy = [] ;
h=cell(1,10961);

for i=1:10961

nett = vl_simplenn_move(net, 'gpu');    
I = gpuArray(images.data(:,:,1,(i-1)*6+1:i*6));
res = vl_simplenn(nett, I ,res,dzdy,'mode','test') ;

h{i} = gather(squeeze(res(end).x));
res=[];
gpuDevice(1);
end