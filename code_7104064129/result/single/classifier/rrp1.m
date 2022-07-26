hh = [h{1} h{2} h{3} h{4} h{5} h{6} h{7} h{8} h{9} h{10}];


q = size(hh);
x = zeros(q(1),q(2));
e = exp(hh);

for i=1:q(2)
    for j= 1:q(1)
        
        prob(j,i) = e(j,i)/sum(e(:,i));
        r(1,i) = find(prob(:,i) == max(prob(:,i)));
         
               
                       
    end
end

for i=1:73
   
    ll(1,i) = mean(l(1,(i-1)*180+1:i*180));
    rr(1,i) = mode(r(1,(i-1)*180+1:i*180));
    
end

size(find((ll-rr)==0))

l = images.labels;

for i=1:4380
   
    ll(1,i) = mean(l(1,(i-1)*60+1:(i-1)*60+60));
    rr(1,i) = mode(r(1,(i-1)*60+1:(i-1)*60+60));
    rr(2,i) = mode(r(1,(i-1)*60+4381:(i-1)*60+4380+60));
    rr(3,i) = mode(r(1,(i-1)*60+4380*2+1:(i-1)*60+4380*2+60));
    
end

for i=1:73
   
    rg(1,i) = mode(rr(:,i));
    
end


a1 = find(rg==1);
a2 = find(rg==2);
a3 = find(rg==3);
a4 = find(rg==4);