clear;
clc;
modelfile = 'nn_model.onnx';

% load NN with CORA tool box
network = neuralNetwork.readONNXNetwork('nn_model.onnx');

% define the normalisation function
means = [68.479297, 69.437695, 0.393321, 0.718784];
SDs   = [52.128595, 52.274740, 2.142253, 1.982537];
normalise = @(x_unnorm) (x_unnorm - means) ./ SDs; 

% random define a test point, and normalise
x_test_unnorm = [50, 60, -2.0, 0.5];
x_test_norm = normalize(x_test_unnorm);

% run model only once (via 'struct'), with CORA
y_output = network.evaluate(x_test_norm', struct);
disp(y_output);


