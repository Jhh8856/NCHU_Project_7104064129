function Y = tee(X,c,dzdy)
%%% 將資料轉為0和1 %%%
[p q r s] = size(X);
h=(squeeze(gather(X)));

cc = de2bin((c),4);

for i=1:s
    for j=1:r
        
       if h(j,i) <= 0.5 
          h(j,i) = 0;
       else
           h(j,i) =1;
       end
    end
end
    

for i=1:s
   for j=1:r
       p(j,i) = abs(h(j,i) - cc(j,i));
   end
end
for i=1:s
    
    pre(1,i) = 8*p(1,i)+4*p(2,i)+2*p(3,i)+p(4,i);
    
       
  
end
%disp(pre);
if nargin <= 2 
    
Y = (0.5*(mean((pre).^2)));
%disp(Y);

else
    




%disp(size(r));

for j=1:s    
    
    %Y(1,j) = abs((r(1,j))*16);
    %Y(2,j) = abs((r(2,j))*8);
    %Y(3,j) = abs((r(3,j))*4);
    %Y(4,j) = abs((r(4,j))*2);
    %Y(5,j) = abs((r(5,j))*1);
    
    Y(1,j) = abs((cc(1,j)-h(1,j))*8);
    Y(2,j) = abs((cc(2,j)-h(2,j))*4);
    Y(3,j) = abs((cc(3,j)-h(3,j))*2);
    Y(4,j) = abs((cc(4,j)-h(4,j))*1);
   % Y(5,j) = abs((r(5,j)-h(5,j))*1);
    
end
%disp(Y)


Y = reshape(Y,size(X)).*dzdy;
%disp((Y));
end