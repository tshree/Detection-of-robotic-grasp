%%this is to train the second pass network with hiddenParams=200



% Main script for training a network for grasp recognition (which can then
% also be used for detection.) Assumes data has already been loaded and
% processed. 
% 
% Pre-trains two hidden layers using the sparse autoencoder algorithm.
% Training for the first layer will include our multimodal regularization
% function (doesn't apply to the second layer since there's no distinction
% in modalities after the first layer). 
%
% Then, trains a classifier on top of the second hidden layer's features
% and back-propagates through the network to arrive at a final network
% trained for recognition. Reports results on both the training and testing
% sets.
%
% Note that minFunc may give you some warnings like:
% "Matrix is close to singular or badly scaled." 
% for the first few iterations. This is OK, and will go away quickly.
%
% Author: Ian Lenz


% Switch this to your path to minFunc
addpath ../minFunc

% Comment this out if you want to use MATLAB's GPU functionality for
% backprop
addpath nogpu/

% Load processed data
load ../data/graspTrainData
load ../data/graspTestData
load ../data/graspModes24


% Train layer 1 with sparse autoencoder
clear opttheta;
runSAEMultiSparse_2;   %% contains no. of features to be learned by network i.e. 200 here 

% Have to scale the input features based on mask (runSAEMultiSparse does
% this for pre-training, but we also need it for computing features and
% backprop.)
trainMask = scaleMaskByModes2(trainMask,trainModes);
trainData = trainData.*trainMask;
testMask = scaleMaskByModes2(testMask,trainModes);
testData = testData.*testMask;

% Scale the first-layer weights to get some desirable statistics in the
% first-layer features. 
W_2= scaleAndBiasWeights(W_2(:,1:size(W_2,2)-1)',trainData,1.5,0.15);

% Compute first-layer features.
trainFeat1 = 1./(1+exp(-[trainData ones(size(trainData,1),1)]*W_2));

% Save first-layer weights.
save ../weights/graspWL1_2 W_2;

% Train second layer similarly based on first-layer features.
clear opttheta;
runSAEBinary_2;

W_2 = scaleAndBiasWeights(W_2(:,1:size(W_2,2)-1)',trainFeat1,1.5,0.15);
save ../weights/graspWL2_2 W_2;
clear trainFeat1;

% Run backpropagation
runBackpropMultiReg;

% Save post-backprop weights
save ../weights/graspWFinal_2 W1 W2 W_class;     %%for network second
