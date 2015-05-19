%Copyright 2015 Ramalho, Ruben
%
%Licensed under the Apache License, Version 2.0 (the "License");
%you may not use this file except in compliance with the License.
%You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
%Unless required by applicable law or agreed to in writing, software
%distributed under the License is distributed on an "AS IS" BASIS,
%WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%See the License for the specific language governing permissions and
%limitations under the License.

%Example code
clear;
clc

rand('state',0);
load('./MNIST.mat');
tic

annealedMomentum = linspace(0.5,0.99,50);

opt = optimNN(    'Iter', 50, ... 
                  'MiniBatchSize', 100, ...
                  'Method','NAG', ...
                  'LearningRate', 0.1, ...
                  'Momentum', annealedMomentum, ...
                  'MaxNorm', 2 ...
             )

nn = setupNN(    'Topology', ...
                 [784, 800, 800, 10], ...
                 'Dropout',[0.2,0.5,0.5,0.5,0.5] ...
            )

%Train the DNN
trained_nn = train_ffnn(opt, nn, Xtrain, ctrain);

pred_test =  predict_ffnn(trained_nn, Xtest);

%Calculate test set accuracy:
fprintf('\nTest Set Accuracy: %f %%\n',mean(double(pred_test == ctest)) * 100);
toc