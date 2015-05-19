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

%Transforms the dataset class's into one hot encoding format.
function [C, n] = onehot(Y)

    classes = unique(Y);
    m = length(Y);
    n = length(classes);
    C = sparse(1:m,Y,1);
end