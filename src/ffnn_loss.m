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

function [J, Grad, m] = ffnn_loss(nn, X, C)
              
    % Setup some useful variables
    [~, m] = size(X);
            
    dropoutMask = [];
    if strcmp(nn.dropout,'yes')
        dropoutMask = initDropoutDistr(nn.tot_layers + 1,nn.topology, nn.dropout_p);
    end
    
    %Feed forward
    [C_hat, sig_activation, activation] = feedforward(nn, X, dropoutMask);
    
    %Back prop, compute gradient for block
    grad = backprop(nn, sig_activation, activation, X, C, dropoutMask);            
    
    %Return Grad in a compressed matter.
    Grad = unroll(grad);
    
    if strcmp(nn.CostSensitive, 'yes')
        %Havent tested this hack for cross entropy, work's fine for
        %loglikelihood tough.
        CostWeights = repmat(nn.CostWeights,1,size(C,2));
        J = nn.cost_function(C_hat, C.*CostWeights);
    else        
        %Default way of computing the cost        
        J = nn.cost_function(C_hat, C);
    end        
    
    function unrolled = unroll(rolled)
        unrolled = zeros(nn.total_weights,1);
        for layer=1:nn.tot_layers
            unrolled((nn.layer_idxs(layer)+1):nn.layer_idxs(layer+1),1) = rolled{layer};
        end
    end
end