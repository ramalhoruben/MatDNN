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


function normalized_nn_params = maxNorm(nn_params, nn, maxNorm)

    idxs = nn.layer_idxs;
    
    normalized_nn_params = zeros(size(nn_params));
    
    for layer=1:nn.tot_layers
        layer_theta = reshape(nn_params(idxs(layer)+1:idxs(layer+1)),...
            nn.topology(layer+1), nn.topology(layer)+1);
        layer_norm = sqrt(sum(layer_theta.^2,2));
        
        tmp = maxNorm*repmat((layer_norm > maxNorm)./layer_norm, 1, nn.topology(layer)+1) ...
            + repmat(layer_norm <= maxNorm, 1, nn.topology(layer)+1);     

        normalized_nn_params(idxs(layer)+1:idxs(layer+1)) = layer_theta(:) .* tmp(:);

    end
end