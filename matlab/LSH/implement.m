clc;
clear all;

load('mnist.mat');
Xtr=full(Xtr);
Xte=full(Xte);
[~,s]=size(Xte);

y1=zeros(1,s);
tic
for i=1:s
    i
    x=Xte(:,i);
    y1(i)=nn_obo(x,Xtr,ytr)-yte(i);
end
t1=toc
a1=1-sum(y1~=0)/10000


T1=lsh(5,5,size(Xtr,1),Xtr);
y2=zeros(1,100);
tic;
for i=1:s
    i
    x=Xte(:,i);
    [nnlsh,~]=lshlookup(x,Xtr,T1,1);
    y2(i)=ytr(nnlsh)-yte(i);
end
t2=toc
a2=1-sum(y2~=0)/10000

