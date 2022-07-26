load net-epoch-40


res = [];
dzdy = [] ;
h=cell(1,3240);

for i=1:3240

nett = vl_simplenn_move(net, 'gpu');    
I = gpuArray(images.data(:,:,1,(i-1)*20+1:i*20));
res = vl_simplenn(nett, I ,res,dzdy,'mode','test') ;

h{i} = gather(squeeze(res(end).x));
res=[];
gpuDevice(1);
end
