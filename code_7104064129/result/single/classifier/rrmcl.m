hh = [h{1} h{2} h{3} h{4} h{5} h{6}];


q = size(hh);
x = zeros(q(1),q(2));
e = exp(hh);

for i=1:q(2)
    for j= 1:q(1)
        
        prob(j,i) = e(j,i)/sum(e(:,i));
        r(1,i) = find(prob(:,i) == max(prob(:,i)));
         
               
                       
    end
end



l = images.labels;
for i=1:16
   
    ll(1,i) = mean(l(1,(i-1)*192+1:(i-1)*192+192));
    rr(1,i) = mode(r(1,(i-1)*192+1:(i-1)*192+192));
    rr(2,i) = mode(r(1,(i-1)*192+56454+1:(i-1)*192+56454+192));
    rr(3,i) = mode(r(1,(i-1)*192+56454*2+1:(i-1)*192+56454*2+192));
    
end

for i=1:31
   
    ll(1,i+76) = mean(l(1,(i-1)*510+1+(16*192):(i-1)*510+510+(16*192)));
    rr(1,i+76) = mode(r(1,(i-1)*510+1+(16*192):(i-1)*510+510+(16*192)));
    rr(2,i+76) = mode(r(1,(i-1)*510+56454+1+(16*192):(i-1)*510+510+56454+(16*192)));
    rr(3,i+76) = mode(r(1,(i-1)*510+56454*2+1+(16*192):(i-1)*510+510+56454*2+(16*192)));
    
end

for i=1:31
   
    ll(1,i+76+152) = mean(l(1,(i-1)*192+1+(16*192+31*510):(i-1)*192+192+(16*192+31*510)));
    rr(1,i+76+152) = mode(r(1,(i-1)*192+1+(16*192+31*510):(i-1)*192+192+(16*192+31*510)));
    rr(2,i+76+152) = mode(r(1,(i-1)*192+56454+1+(16*192+31*510):(i-1)*192+192+56454+(16*192+31*510)));
    rr(3,i+76+152) = mode(r(1,(i-1)*192+56454*2+1+(16*192+31*510):(i-1)*192+192+56454*2+(16*192+31*510)));
    
end

for i=1:62
   
    ll(1,i+76+152+152) = mean(l(1,(i-1)*510+1+(16*192+31*510+31*192):(i-1)*510+510+(16*192+31*510+31*192)));
    rr(1,i+76+152+152) = mode(r(1,(i-1)*510+1+(16*192+31*510+31*192):(i-1)*510+510+(16*192+31*510+31*192)));
    rr(2,i+76+152+152) = mode(r(1,(i-1)*510+56454+1+(16*192+31*510+31*192):(i-1)*510+510+56454+(76*192+31*510+31*192)));
    rr(3,i+76+152+152) = mode(r(1,(i-1)*510+56454*2+1+(16*192+31*510+31*192):(i-1)*510+510+56454*2+(76*192+31*510+31*192)));
    
end

for i=1:140
    
    rg(1,i) = round(mean(rr(:,i)));
    
end
g1 = find(rg==1);
g2 = find(rg==2);
g3 = find(rg==3);
g4 = find(rg==4);