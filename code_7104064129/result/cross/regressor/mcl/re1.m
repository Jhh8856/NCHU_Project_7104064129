for i=1:35691
   
    hh(:,(i-1)*2+1:i*2) = h{1,i};
    
end


q = size(hh);


for i=1:q(2)
    for j= 1:q(1)
        
     if hh(j,i) < 0.5
        x(j,i) = 0;
     else
         x(j,i) = 1;
     end
                       
    end 
end

xx=ones(1,71382);
for i=1:71382
              
       xx(1,i) = 8*x(1,i)+4*x(2,i)+2*x(3,i)+1*x(4,i);
        
end

for i=1:21
   
    ll(1,i) = mean(l(1,(i-1)*192+1:(i-1)*192+192));
    rr(1,i) = mode(r(1,(i-1)*192+1:(i-1)*192+192));
    
end

for i=1:40
   
    ll(1,i+21) = mean(l(1,(i-1)*510+1+(21*192):(i-1)*510+510+(21*192)));
    rr(1,i+21) = mode(r(1,(i-1)*510+1+(21*192):(i-1)*510+510+(21*192)));
    
end

for i=1:40
   
    ll(1,i+61) = mean(l(1,(i-1)*192+1+(21*192+40*510):(i-1)*192+192+(21*192+40*510)));
    rr(1,i+61) = mode(r(1,(i-1)*192+1+(21*192+40*510):(i-1)*192+192+(21*192+40*510)));
    
end

for i=1:75
   
    ll(1,i+101) = mean(l(1,(i-1)*510+1+(21*192+40*510+40*192):(i-1)*510+510+(21*192+40*510+40*192)));
    rr(1,i+101) = mode(r(1,(i-1)*510+1+(21*192+40*510+40*192):(i-1)*510+510+(21*192+40*510+40*192)));
    
end




r = corr(y',lf','type','pearson');
rmse = sqrt(mean(((y) - lf).^2));