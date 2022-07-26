function [net]  = firstsmx(varargin)
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;


rng('default');
rng(0) ;

lr = [.1 2] ;
f=0.00001;
f1=0.001;
net.layers = {} ;
                      
net.layers{end+1} = struct('name', 'conv1', ...
                           'type', 'conv', ...
                           'weights', {{f*randn(3,3,1,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
%net.layers{end+1} = struct('type', 'sigmoid') ;                       
net.layers{end+1} = struct('name','relu1', ...
                           'type', 'relu');      
                       
net.layers{end+1} = struct('name', 'pool1', ...
                           'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride',1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('name', 'conv2', ...
                           'type', 'conv', ...
                           'weights', {{f1*randn(3,3,32,64, 'single'), zeros(1,64, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'sigmoid') ;                      
 net.layers{end+1} = struct('name', 'relu2', ...
                            'type', 'relu') ;         
                                              
net.layers{end+1} = struct('name', 'pool2', ...
                           'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 0) ;    
                       
 net.layers{end+1} = struct('name', 'conv3', ...
                            'type', 'conv', ...
                            'weights', {{f*randn(3,3,64,128, 'single'), zeros(1, 128, 'single')}}, ...
                            'stride', 1, ...
                            'pad', 0) ;
 
 net.layers{end+1} = struct('name','relu3', ...
                            'type', 'relu');                       
                       
                           
 net.layers{end+1} = struct('name', 'pool3', ...
                            'type', 'pool', ...
                            'method', 'max', ...
                            'pool', [3 3], ...
                            'stride',1, ...
                            'pad', 0) ;           
                   
 net.layers{end+1} = struct('name', 'conv4', ...
                            'type', 'conv', ...
                            'weights', {{f1*randn(3,3,128,160, 'single'), zeros(1, 160, 'single')}}, ...
                            'stride', 1, ...
                            'pad', 0) ;
 
 net.layers{end+1} = struct('name','relu4', ...
                            'type', 'relu');                         
                        
 net.layers{end+1} = struct('name', 'pool4', ...
                            'type', 'pool', ...
                            'method', 'max', ...
                            'pool', [3 3], ...
                            'stride',1, ...
                            'pad', 0) ;  
                        
   
                        
   net.layers{end+1} = struct('name', 'conv5', ...
                            'type', 'conv', ...
                            'weights', {{f1*randn(3,3,160,192, 'single'), zeros(1, 192, 'single')}}, ...
                            'stride', 1, ...
                            'pad', 0) ;
                        
 net.layers{end+1} = struct('name','relu4', ...
                            'type', 'relu');                         
                                                                    
 net.layers{end+1} = struct('name', 'pool5', ...
                            'type', 'pool', ...
                            'method', 'max', ...
                            'pool', [3 3], ...
                            'stride',1, ...
                            'pad', 0) ;  
                        
  net.layers{end+1} = struct('name', 'conv6', ...
                            'type', 'conv', ...
                            'weights', {{f1*randn(3,3,192,224, 'single'), zeros(1,224, 'single')}}, ...
                            'stride', 1, ...
                            'pad', 0) ;
                        
 net.layers{end+1} = struct('name','relu4', ...
                            'type', 'relu');                         
                                                                    
 net.layers{end+1} = struct('name', 'pool6', ...
                            'type', 'pool', ...
                            'method', 'max', ...
                            'pool', [3 3], ...
                            'stride',1, ...
                            'pad', 0) ;                          
                         
net.layers{end+1} = struct('name', 'fucon1', ...
                           'type', 'conv', ...
                           'weights', {{f1*randn(40,40,224,256, 'single'),  zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'sigmoid') ;                      
 net.layers{end+1} = struct('name','relu4', ...
                            'type', 'relu');  
                       
 net.layers{end+1} = struct('name', 'fucon2', ...
                           'type', 'conv', ...
                           'weights', {{f1*randn(1,1,256,4, 'single'),  zeros(1,4,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                            'pad', 0) ;
                        
 net.layers{end+1} = struct('name','relu5', ...
                            'type', 'relu');  
                        

                        
                        
 net.layers{end+1} = struct('type','softmaxloss');
                             





if opts.batchNormalization
  net = insertBnorm(net,1) ;  
  net = insertBnorm(net,5) ;
  net = insertBnorm(net,9) ;
  net = insertBnorm(net,13) ;
  net = insertBnorm(net,17) ;
  net = insertBnorm(net,21) ;
end

% Meta parameters
net.meta.inputSize = [64 64] ;
net.meta.trainOpts.weightDecay = 1;
net.meta.trainOpts.learningRate = 0.000001;
net.meta.trainOpts.numEpochs = 2;
net.meta.trainOpts.batchSize = 50;
net.meta.sets = {'train', 'val'} ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end
% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;