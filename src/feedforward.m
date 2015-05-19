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

%Standard feedforward algorithm for nn, with dropout capabilitiess

function [c_hat, sig_activation, activation] = feedforward(nn, X_mini_batch, dropoutMask)
       
       fst_layer = 1;
       lst_layer = nn.tot_layers;
       
       activation = cell(nn.tot_layers+1,1);
       sig_activation = cell(nn.tot_layers+1,1);
       
       %Feed-forward
       bias = ones(1,size(X_mini_batch,2));
       sig_activation{fst_layer} = X_mini_batch(:,:);

       for layer=2:lst_layer
           
           Y = [bias ; sig_activation{layer-1}];
           %If dropout is enabled apply dropout mask
           if ~isempty(dropoutMask)
               %Improving neural networks by preventing co-adaptation of
               %feature detectors, Hinton et al (2012).
               Y = Y .* [bias; repmat(dropoutMask{layer-1},1,size(Y,2))];
           end
           
           activation{layer} = decompress(nn, layer-1) * Y;           
           sig_activation{layer} = nn.activation_function(activation{layer});
       end
       
       Y = [bias ; sig_activation{lst_layer}];
       if ~isempty(dropoutMask)
            Y = Y .* [bias; repmat(dropoutMask{lst_layer},1,size(Y,2))];
       end
       
       activation{lst_layer+1} = decompress(nn, lst_layer) * Y;
       
       if sum(sum(isnan(activation{lst_layer+1}))) > 0
           decompress(nn, lst_layer)
       end
       
       sig_activation{lst_layer+1} = nn.activation_function_output(activation{lst_layer+1});
       c_hat = sig_activation{lst_layer+1};
end