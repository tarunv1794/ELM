function [ accuracy ] = testing( X_test, InputWeight, OutputWeight, y_test, m_test, lambda, functype, threshold )
%% =========== Calculating Testing Set Accuracy =============
tempH = ([ones(m_test, 1) X_test] * InputWeight');
H = choosefunc(lambda, functype, tempH, threshold);
H = [ones(m_test, 1) H];

Ytest = H * OutputWeight;
[~, p] = max(Ytest, [], 2);
cnt = 0;

for i=1:m_test
    if(p(i) == y_test(i))
        cnt = cnt + 1;
    end
end

accuracy = cnt/m_test; 
end