function b = de2bin(d,c)

[p q]=size(d);

%disp(q)

if c==3
   b = zeros(3,q);
    for i=1:q    
        if d(i)>=4
           b(1,i) = 1;
           d(i) = d(i)-4;
        end
        if d(i)>=2
           b(2,i) = 1;
           d(i) = d(i)-2;
        end
        if d(i)>=1
           b(3,i)=1;
        end   
    end



elseif c==4
   b = zeros(4,q);
    for i=1:q
    

        if d(i)>=8
           b(1,i) = 1;
           d(i) = d(i)-8;
        end
        if d(i)>=4
           b(2,i) = 1;
           d(i) = d(i)-4;
        end
        if d(i)>=2
           b(3,i) = 1;
           d(i) = d(i)-2;
        end
        if d(i)>=1
           b(4,i)=1;
        end   
    end

elseif c==5
    b=zeros(5,q);


    for i=1:q
    
    if d(i)>=16
    b(1,i) = 1;
    d(i) = d(i)-16;
    end
    if d(i)>=8
    b(2,i) = 1;
    d(i) = d(i)-8;
    end
    if d(i)>=4
    b(3,i) = 1;
    d(i) = d(i)-4;
    end
    if d(i)>=2
    b(4,i) = 1;
    d(i) = d(i)-2;
    end
    if d(i)>=1
    b(5,i)=1;            
    end
    
    end
    
end
    

