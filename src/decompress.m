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

%Given the condensed params of a nn, topology and layer we return
%the weight matrix for the specified layer.

function theta = decompress(nn, layer)
       
    theta = reshape(nn.layer_weights{layer},...
        nn.topology(layer+1), nn.topology(layer)+1);    
end