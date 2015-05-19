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

function grad = backprop(nn, sig_activation, activation, X, C, dropoutMask)
        
        topology = nn.topology;
        
        %Auxiliar variables..
        lst_layer = length(topology);

        error = cell(length(topology),1);
        grad = cell(length(topology),1);

        m = size(X,2);
        bias = ones(1,m);
        
        if strcmp(nn.CostSensitive, 'yes')
            CostWeights = repmat(nn.CostWeights,1,size(C,2));
            error{lst_layer} = CostWeights.*(sig_activation{lst_layer} - C);
        else
            error{lst_layer} = (sig_activation{lst_layer} - C);
        end
        
        %Back-prop
        layer_grad = error{lst_layer} * [bias;sig_activation{lst_layer-1}]';  

        if ~isempty(dropoutMask)
            %Improving neural networks by preventing co-adaptation of
            %feature detectors, Hinton et al (2012).
            layer_grad = layer_grad .* repmat(dropoutMask{lst_layer} > 0, ...
                1,size(layer_grad,2));                
        end
        grad{lst_layer-1} = layer_grad(:)/m;

        for layer=(lst_layer-1):-1:2

            error{layer} = der_error(layer, nn, error, activation);
                        
            layer_grad = error{layer} * [bias;sig_activation{layer-1}]';   
            if ~isempty(dropoutMask)
                %Improving neural networks by preventing co-adaptation of
                %feature detectors, Hinton et al (2012).
                layer_grad = layer_grad .* repmat(dropoutMask{layer} > 0, ...
                    1,size(layer_grad,2));                
            end

            grad{layer-1} = layer_grad(:)/m;
        end    
        
    function E = der_error(layer, nn, errors, activations)

%        bias = ones(1,size(activations{layer},2));
        e = decompress(nn ,layer)'*errors{layer+1} .* ...
                    nn.activation_derivative([bias;activations{layer}]);
        E = e(2:end,:); 
    end
end

