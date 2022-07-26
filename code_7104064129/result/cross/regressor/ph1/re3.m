for i=1:1560
   
    hh(:,(i-1)*12+1:i*12) = h{1,i};
    
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

xx=ones(1,18720);
for i=1:18720
              
       xx(1,i) = 8*x(1,i)+4*x(2,i)+2*x(3,i)+1*x(4,i);
        
end

for i=1:18720/180
    
    y(1,i) = mean(xx((i-1)*180+1:i*180));
    lf(1,i) = mean(l((i-1)*180+1:i*180));
    
end

r = corr(y',lf','type','pearson');
rmse = sqrt(mean(((y+30) - lf).^2));