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


% See: Xavier and  Bengio , "Understanding the difficulty of training deep feedforward neural
% networks"

function init = log_initWeights(layer_sizes, layer, topology)

    e = 4*sqrt(6/(layer_sizes(1,layer-1) + topology(1,layer+1)));
    init = rand(layer_sizes(layer),1)*2*e - e;
end
