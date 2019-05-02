
function T = lsh(l,k,d,x)

b=inf;
range=[zeros(1,d);ones(1,d)];
ind=1:size(x,2);

% create the LSH functions
Is = lshfunc(l,k,d);

% index the data in X using these LSH functions
T = lshprep(Is,b);

if (~isempty(x))
  T = lshins(T,x,ind);
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I = lshfunc(l,k,d)
exclude=[];
w=[];
range =[zeros(1,d);ones(1,d)];
  
% set up interval width if necessary
if (isempty(w))
    w=-16;
end
if (w < 0)
    %estimate of the range of the projection
    limits = max(abs(range(1,:)),abs(range(2,:)));
    rangeAct=mean(diff([-limits; limits]*2*sqrt(d)));
    n=abs(w);
    w = rangeAct/n;
end

for j=1:l
    % there are k functions determined by random vectors + random shifts
    % hash key: floor((A'*x-b)/W)
    I(j).W = w;
    I(j).A = randn(d,k);
    I(j).b = unifrnd(0,w,1,k);
    I(j).k = k;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function T = lshprep(Is,B)

l = length(Is);  % number of hash key
k = Is(1).k;

for j=1:l
  T(j).I = Is(j);
  T(j).B = B;
  T(j).count = 0;
  T(j).buckets = [];
  T(j).Index = {};

  % set up secondary hash table for buckets
  % max. index can be obtained by running lshhash on max. bucket
  T(j).bhash = cell(lshhash(ones(1,k)*255),1);
  
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function T = lshins(T,x,ind)
if (nargin < 3 | isempty(ind))
  ind=1:size(x,2);
end

% insert in each table
for j=1:length(T)
  
  % the number of buckets before new data
  oldBuckets=size(T(j).buckets,1);
  
  % find, for each data point, the corresp. bucket
  buck = findbucket(x,T(j).I);
  % now x(:,n) goes to bucket with key uniqBuck(bID(n))
    
  [uniqBuck,ib,bID] = unique(buck,'rows');
  keys = lshhash(uniqBuck);
  
  T(j).buckets=[T(j).buckets; zeros(length(ib),T(j).I.k,'uint8')];
  
  newBuckets=0;
  
  for b=1:length(ib)
    % find which data go to bucket uniqBuck(b)
    thisBucket = find(bID==bID(ib(b)));
    
    % find out if this bucket already has anything
    % first, which bucket is it?
    ihash = T(j).bhash{keys(b)}; % possible matching buckets
    if (isempty(ihash)) % nothing matches
      isb = [];
    else % may or may not match
      isb = ihash(find(all(bsxfun(@eq,uniqBuck(b,:),T(j).buckets(ihash,:)),2)));
    end
    
    
    if (~isempty(isb)) 
      % adding to an existing bucket.
      oldcount=length(T(j).Index{isb}); 
      newIndex = [T(j).Index{isb}  ind(thisBucket)];
    else
      % creating new bucket
      newBuckets=newBuckets+1;
      oldcount=0;
      isb = oldBuckets+newBuckets;
      T(j).buckets(isb,:)=uniqBuck(b,:);
      T(j).bhash{keys(b)} = [T(j).bhash{keys(b)}; isb];
      newIndex = ind(thisBucket);
    end
    
    %put this into the table
    T(j).Index{isb}= newIndex;
    % update distinct element count
    T(j).count = T(j).count + length(newIndex)-oldcount;
    
  end
  % we may not have used all of the allocated bucket space
  T(j).buckets=T(j).buckets(1:(oldBuckets+newBuckets),:);
  fprintf(2,'Table %d adding %d buckets (now %d)\n',j,newBuckets,size(T(j).buckets,1));
  fprintf(2,'Table %d: %d elements\n',j,T(j).count);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function v =findbucket(x,I)
v = floor((double(x)'*I.A - repmat(I.b,size(x,2),1))/I.W);
v = uint8(v+128);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hkey,hpos] = lshhash(keys)
P = [1 2 5 11 17 23 31 41 47 59];

[n,m]=size(keys);
M = min(length(P),m);

hpos = zeros(1,M); % indices of positions used to hash
for i=1:M
  if (mod(i,2)==1)
    hpos(i) = (i+1)/2;
  else
    hpos(i) = m-(i/2)+1;
  end
end

hkey = sum(bsxfun(@times,double(keys(:,hpos)),P(1:M)),2)+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [kNN,num] = lshlookup(x0,x,T,k)
distfun='lpnorm';
distargs={2};

r=inf;
sel='best';
f=[];
fargs=[];
l = length(T);
kNN=[];

% find the union of buckets in all tables that match query
for j=1:l
  % look up T_j
  % buck is the # of bucket in T{j}
  buck = findbucket(x0,T(j).I);
  % find the bucket in j-th table
  key = lshhash(buck);
  ihash = T(j).bhash{key}; % possible matching buckets
  if (~isempty(ihash)) % nothing matches
    b = ihash(find(all(bsxfun(@eq,buck,T(j).buckets(ihash,:)),2)));
    if (~isempty(b))
      kNN = [kNN T(j).Index{b}];
    end
  end
end

% delete duplicates
[kNN,iu]=unique(kNN);
num = length(kNN);

% now iNN has the collection of candidate indices 
% we can start examining them

if (~isempty(kNN))
  
  if (strcmp(sel,'best'))

    D=feval(distfun,x0,Xsel(x,kNN),distargs{:});
    [dist,sortind]=sort(D);
    ind = find(dist(1:min(k,length(dist)))<=r);
    kNN=kNN(sortind(ind));
    
  else % random
    
    rp=randperm(num);
    choose=[];
    for i=1:length(rp)
      d = feval(distfun,x0,Xsel(x,kNN(rp(i))),distargs{:});
      if (d <= r)
	choose = [choose kNN(rp(i))];
	if (length(choose) == k)
	  break;
	end
      end
    end
    kNN = choose;
  end
  
end


%%%%%%%%%%%%%%%%%%%%%%%%55 
function x=Xsel(X,ind)
% x=Xsel(X,ind)
% selects (i.e. collects) columns of cell array X
% (automatically determining the class, and looking for each column in
% the right cell.)

if (~iscell(X))
  x=X(:,ind);
  return;
end

d=size(X{1},1);

if (strcmp(class(X{1}),'logical'))
  x=false(d,length(ind));
else
  x=zeros(d,length(ind),class(X{1}));
end
sz=0; % offset of the i-th cell in X
collected=0; % offset within x
for i=1:length(X)
  thisCell=find(ind > sz & ind <= sz+size(X{i},2));
  if (~isempty(thisCell))
    x(:,thisCell)=X{i}(:,ind(thisCell)-sz);
  end
  collected=collected+length(thisCell);
  sz=sz+size(X{i},2);      
end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%