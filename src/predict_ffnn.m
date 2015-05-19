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


%predict_ffnn, predicts the label of an input given a trained neural network
%   [c_hat, p] = predict_ffnn(nn_params, topology, X) outputs the predicted class of X and 
%   class probabilities given the trained weights of a neural network.

function [c_hat, p] = predict_ffnn(nn, X)

    m = size(X,1);
    blocks = ceil(m/10000);
    idxs = [0, cumsum(10000*ones(1,blocks))];
    if mod(size(X,1),10000) ~= 0
        idxs(blocks+1) = idxs(blocks)+ mod(size(X,1), 10000);        
    end
    
    p = zeros(m,nn.num_classes);
    c_hat = zeros(m,1);
    %Split the operation in blocks, this is hack for the operation to fit in memory
    %essencial for larger dataset's. (expect more time however)
    for b=1:blocks
        block = X((idxs(b)+1):idxs(b+1),:)';
        p((idxs(b)+1):idxs(b+1),:) = feedforward(nn, block,[])';
        [~, tmp] = max(p((idxs(b)+1):idxs(b+1),:),[],2);
        c_hat((idxs(b)+1):idxs(b+1),1) = tmp;
    end
end