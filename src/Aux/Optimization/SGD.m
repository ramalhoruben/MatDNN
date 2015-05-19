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

function [J, Update, empty, empty2]= SGD(nn, options, X, Y, epoch)
    
    empty = 0;
    empty2 = 0;
    
    [J, grad] = ffnn_Regularizer(nn, options, X, Y);

    if isscalar(options.LearningRate)
        %Constant Learning Rate
        Update = options.LearningRate*grad;            
    elseif isvector(options.LearningRate)
        %Annealed Learning Rate
        Update = options.LearningRate(epoch)*grad;
    end

end