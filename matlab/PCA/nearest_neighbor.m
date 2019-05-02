%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the average cat
%and dog vectors. It also takes in a test data matrix Xtest and 
%produces a vector of label guesses yguess. Each guess is found
%by searching through Xtrain to find the closest row, and then 
%outputting its label.
function yguess = nearest_neighbor(Xtrain,ytrain,Xtest)

n_train = size(Xtrain, 1);
yguess = [];

%Going through each row of Xtest
for j = 1:size(Xtest, 1)
   currMin = Inf;
   i_closest = 1;
   
   %Current row of Xtest
   x = Xtest(j, :);
   
   %Search through Xtrain
   for i = 1:n_train
       %New row of Xtrain
       x_train_i = Xtrain(i, :);
       
       %Distance from current row of Xtest with new row of Xtrain
       currNorm = norm(x - x_train_i, 2);
       
       %If distance is less than currMin, replace currMin and the index
       if currNorm < currMin
           currMin = currNorm;
           i_closest = i;
       end
   end
   
   %Append label corresponding to i_closest to yguess
   yguess = [yguess; ytrain(i_closest)];
end