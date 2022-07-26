function [net, info] = cnn_reg4(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile('D:\CNN\matconvnet-1.0-beta20', 'matlab', 'vl_setupnn.m')) ;
 


opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ; 
  



opts.expDir = fullfile(vl_rootnn, 'data\test\classification\binary\0519\sfm-4g\sco\g4') ;
[opts, varargin] = vl_argparse(opts, varargin) ;
 

opts.imdbPath = fullfile(opts.expDir, 'im4.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------


net = reg4('networkType', opts.networkType);

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
end

%net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end 

 [net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2));


% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
     fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
 function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
 images = imdb.images.data(:,:,:,batch) ;
 labels = imdb.images.labels(:,batch) ;


% --------------------------------------------------------------------
 function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
 images_src_1 = imdb.images.data_src1(:,:,:,batch) ;
 images_src_2 = imdb.images.data_src2(:,:,:,batch) ;
 images_src_3 = imdb.images.data_src3(:,:,:,batch) ;
 labels = imdb.images.labels(1,batch) ;
 
 if rand >0.5
     images_src_1 = fliplr(images_src_1);
     images_src_2 = fliplr(images_src_2);
     images_src_3 = fliplr(images_src_3);
 end
 if opts.numGpus > 0
  images_src_1 = gpuArray(images_src_1) ;
  images_src_2 = gpuArray(images_src_2) ;
  images_src_3 = gpuArray(images_src_3) ;
 end
  inputs = {'input_1', images_src_1, 'label', labels, 'input_2' , images_src_2 , 'label' , 'labels', 'input_3' , images_src_3 , 'label' , 'labels'} ;


