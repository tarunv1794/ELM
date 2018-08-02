function [maxmulti, indexmulti, maxsingle, indexsingle] = ELMmain(input_layer_size, hidden_layer_size, num_labels, epsilon_init, lambda, threshold)
%% Setting up the parameters
accuracy_single = zeros(2,10);
accuracy_multi  = zeros(2,10);

avg_accuracysingle = zeros(2,1000);
avg_accuracymulti  = zeros(2,1000);

%% =========== Part 1: Loading Data =============

load iris.dat;

for b=1:1000
    for a=1:10
        X_test = [iris((5*(a-1))+1:5*a,1:4);iris((5*(a-1))+51:50+5*a,1:4);iris((5*(a-1))+101:100+5*a,1:4);];
        y_test = [iris((5*(a-1))+1:5*a,5);iris((5*(a-1))+51:50+5*a,5);iris((5*(a-1))+101:100+5*a,5);];
        m_test = 15;

        X_train = [iris(1:(5*(a-1)),1:4);iris(5*a+1:50,1:4);iris(51:(5*(a-1)+50),1:4);iris(5*a+51:100,1:4);iris(101:(5*(a-1)+100),1:4);iris(5*a+101:150,1:4);];
        y_train = [iris(1:(5*(a-1)),5);iris(5*a+1:50,5);iris(51:(5*(a-1)+50),5);iris(5*a+51:100,5);iris(101:(5*(a-1)+100),5);iris(5*a+101:150,5);];
        m_train = 135;

%% =========== Part 2: Transforming output vector =============
        y_transform = zeros(m_train,num_labels);

        for i=1:m_train
            for j=1:num_labels
                if(y_train(i) == j)
                    y_transform(i,j) = 1;
                end
            end
        end

%% =========== Part 3: Randomly Intializing Input Weights =============
        [ InputWeight ] = randInitialization( hidden_layer_size, input_layer_size, epsilon_init );

%% =========== Part 4: Single Threshold Implementation =============
        functype = 1;
        [accuracy_single(1,a), OutputWeight] = training( X_train, InputWeight, y_transform, y_train, m_train, lambda, functype, threshold );
        [accuracy_single(2,a)] = testing ( X_test,  InputWeight, OutputWeight, y_test , m_test , lambda, functype, threshold );
    
%% =========== Part 5: Multi Threshold Implementation =============
        functype = 2;
        [accuracy_multi(1,a), OutputWeight_multi] = training( X_train, InputWeight, y_transform, y_train, m_train, lambda, functype, threshold );
        [accuracy_multi(2,a)] = testing ( X_test,  InputWeight, OutputWeight_multi, y_test , m_test , lambda, functype, threshold );
    end
    avg_accuracysingle(1,b) = (sum(accuracy_single(1,:)))/a;
    avg_accuracysingle(2,b) = (sum(accuracy_single(2,:)))/a;
    avg_accuracymulti (1,b) = (sum(accuracy_multi(1,:)))/a;
    avg_accuracymulti (2,b) = (sum(accuracy_multi(2,:)))/a;
end

[maxmulti, indexmulti] = max(avg_accuracymulti, [], 2);
[maxsingle, indexsingle] = max(avg_accuracysingle, [], 2);
