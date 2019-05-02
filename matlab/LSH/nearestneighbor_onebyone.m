function y=nn_obo(x,X,Y)
[~,s]=size(X);

d=zeros(s,1);

for i=1:s
    d(i)=norm(x-X(:,i));
end
[~,index]=min(d);
y=Y(index);
