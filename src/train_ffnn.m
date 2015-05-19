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

%Trains a multiclass feed-forward neural network classifiers

function [trained_nn, JTrain, JValid] = train_ffnn(opt, nn, X, c)
            
    %Initialization, nn setup.
    if ~exist('opt', 'var') && ~isempty(opt)
        error('Must supply options.');
    end
    
    %Everything is column oriented for performance.
    X = X';
    C = onehot(c)';
        
    [trained_nn, JTrain, JValid] = gradientDescent(nn, opt, X, C);
                
end