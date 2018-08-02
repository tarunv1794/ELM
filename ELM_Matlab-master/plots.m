%% Initialization
clear ; close all; clc

input_layer_size = 4;
num_labels = 3;
lambda = 20;

maxmulti    = zeros(2,10);
maxsingle   = zeros(2,10);
indexmulti  = zeros(2,10);
indexsingle = zeros(2,10);

% =========== Part 1: Number of Hidden Layers Vs Accuracy =============

hidden_layer_size = (5:5:50);
epsilon_init      = 0.4;
threshold         = 20;

for i=1:10
    [maxmulti(:,i), indexmulti(:,i), maxsingle(:,i), indexsingle(:,i)] = ELMmain(input_layer_size, hidden_layer_size(i), num_labels, epsilon_init, lambda, threshold);
end

figure

plot(hidden_layer_size,maxmulti(2,:),hidden_layer_size,maxsingle(2,:),'--');

title('Size of Hidden Layers Vs Accuracy');
xlabel('hiddenlayersize');
ylabel('maximumaverageaccuracy');

legend('Multi', 'Single');
%% =========== Part 2: Epsilon Vs Accuracy =============

hidden_layer_size = 5;
epsilon_init      = (0.2:0.2:2);
threshold         = 20;

for i=1:10
    [maxmulti(:,i), indexmulti(:,i), maxsingle(:,i), indexsingle(:,i)] = ELMmain(input_layer_size, hidden_layer_size, num_labels, epsilon_init(i), lambda, threshold);
end

figure

plot(epsilon_init,maxmulti(2,:),epsilon_init,maxsingle(2,:),'--');

title('Epsilon Vs Accuracy');
xlabel('epsilon');
ylabel('maximumaverageaccuracy');

legend('Multi', 'Single');
%% =========== Part 3: Threshold Vs Accuracy =============

hidden_layer_size = 5;
epsilon_init      = 0.4;
threshold         = (4:3:31);

for i=1:10
    [maxmulti(:,i), indexmulti(:,i), maxsingle(:,i), indexsingle(:,i)] = ELMmain(input_layer_size, hidden_layer_size, num_labels, epsilon_init, lambda, threshold(i));
end

figure

plot(threshold,maxmulti(2,:),threshold,maxsingle(2,:),'--');

title('Threshold Vs Accuracy');
xlabel('threshold');
ylabel('maximumaverageaccuracy');

legend('Multi', 'Single');