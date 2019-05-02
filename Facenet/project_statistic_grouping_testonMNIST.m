clc;
clear all;
load('mnist.mat');
Xtr=full(Xtr);
Xte=full(Xte);

group_number1_train=zeros(1,60000);
group_number2_train=zeros(1,60000);
group_number_test=zeros(1,10000);


for m=1:60000
    a=zeros(1,28);
    for i=1:28
        for j=1:28
            if Xtr((i-1)*28+j,m)>0.5
                a(i) = a(i) + 1;
            end
        end
    end
    [~,largest_index]=max(a);
    group_number1_train(m)=largest_index;
    a(largest_index)=0;
    [~,second_largest_index]=max(a);
    group_number2_train(m)=second_largest_index;
end

for m=1:10000
    b=ones(1,60000);
    b=100*b;
    a=zeros(1,28);
    for i=1:28
        for j=1:28
            if Xte((i-1)*28+j,m)>0.5
                a(i) = a(i) + 1;
            end
        end
    end
    [~,index]=max(a);
    group_number_test(m)=index;
end
    
correct_time=0;
mistake_time=0;
for m=1:10000
    b=ones(1,60000);
    b=100*b;
    for k=1:60000
        if group_number1_train(k)==group_number_test(m)
            b(k)=norm(Xtr(:,k)-Xte(:,m));
        end
        if group_number2_train(k)==group_number_test(m)
            b(k)=norm(Xtr(:,k)-Xte(:,m));
        end
    end
    [~,index]=min(b);
    if yte(m)==ytr(index)
        correct_time=correct_time+1;
    end
    if yte(m)~=ytr(index)
        mistake_time=mistake_time+1;
    end
end
                
        