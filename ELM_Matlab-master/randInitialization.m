function [ InputWeight ] = randInitialization( hidden_layer_size, input_layer_size, epsilon_init )
InputWeight = rand(hidden_layer_size, 1 + input_layer_size) * 2 * epsilon_init - epsilon_init;
end

