hh = [h{1} h{2} h{3} h{4} h{5} h{6} h{7} h{8} h{9} h{10}];


q = size(hh);
x = zeros(q(1),q(2));

for i=1:q(2)
    for j= 1:q(1)
        
     if h(j,i) >= 0.5
        x(j,i) = 1;
     end
                       
    end 
end

xx=ones(1,size(a,2));

for i=1:size(a,2)
              
       xx(1,i) = 8*x(1,i)+4*x(2,i)+2*x(3,i)+1*x(4,i);
         
       
end

for i=1:size(a,2)/60
    
    y(1,i) = mean(xx((i-1)*60+1:i*60));
    y(2,i) = mean(xx((i-1)*60+size(a,2)/5+1:i*60+size(a,2)/5));
    y(3,i) = mean(xx((i-1)*60+size(a,2)*2/5+1:i*60+size(a,2)*2/5));
    y(4,i) = mean(xx((i-1)*60+size(a,2)*3/5+1:i*60+size(a,2)*3/5));
    y(5,i) = mean(xx((i-1)*60+size(a,2)*4/5+1:i*60+size(a,2)*4/5));
    
    lf(1,i) = mean(l((i-1)*60+1:i*60));
    
end

for i=1:124
    yy(1,i) = (y(1,i) + y(2,i) + y(3,i) + y(4,i) + y(5,i))/5;
end

lf = lf(1,1:size(a,2)/5);

r1 = corr(y',lf','type','spearman');
r2 = corr(y',lf','type','pearson');
rmse = sqrt(mean((yy - lf).^2));