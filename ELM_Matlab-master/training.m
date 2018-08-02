function [ accuracy, OutputWeight ] = training( X_train, InputWeight, y_transform, y_train, m_train, lambda, functype, threshold )

%% =========== Training =============
tempH = ([ones(m_train, 1) X_train] * InputWeight');
H = choosefunc(lambda, functype, tempH, threshold);
H = [ones(m_train, 1) H];
OutputWeight = pinv(H) * y_transform;

%% =========== Calculating Training Set Accuracy =============
Ytrain = H * OutputWeight;
[~, p] = max(Ytrain, [], 2);
cnt = 0;

for i=1:m_train
    if(p(i) == y_train(i))
        cnt = cnt + 1;
    end
end

accuracy = cnt/m_train; 
end