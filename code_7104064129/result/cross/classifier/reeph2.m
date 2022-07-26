for i=1:3240
   
    hh(:,(i-1)*20+1:i*20) = h{1,i};
    
end


q = size(hh);
e = exp(hh);

for i=1:q(2)
    for j= 1:q(1)
        
        prob(j,i) = e(j,i)/sum(e(:,i));
        r(1,i) = find(prob(:,i) == max(prob(:,i)));
         
               
                       
    end
end


%%%%%%%%%%%%%
l = images.labels;

for i=1:360*3
   
    ll(1,i) = mean(l(1,(i-1)*60+1:(i-1)*60+60));
    rr(1,i) = mode(r(1,(i-1)*60+1:(i-1)*60+60));
    rr(2,i) = mode(r(1,(i-1)*60+21601:(i-1)*60+21600+60));
    rr(3,i) = mode(r(1,(i-1)*60+21600*2+1:(i-1)*60+21600*2+60));
    
end

for i=1:360
   
    rg(1,i) = round(mean(rr(:,i)));
    
end


a1 = find(rg==1);
a2 = find(rg==2);
a3 = find(rg==3);
a4 = find(rg==4);