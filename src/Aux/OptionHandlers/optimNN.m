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


function obj = optimNN(varargin)
    
    %Default options
    obj.Iter = 50;
    obj.Regularizer = 'None';
    
    allfields = {'Iter';'MiniBatchSize';'Method';'LearningRate';'Momentum';'MaxNorm'; ...
                'Saturation';'Decay';'e';'Regularizer';'regLambda';'validSet'};

    for i=1:nargin
        arg = varargin{i};
        for j=1:size(allfields)
           if strcmpi(arg,allfields{j})
               obj.(arg) = varargin{i+1};
               i = i+1;
           end
        end
    end
    
    if ~isfield(obj, 'Iter') 
        warning('The number of iterations should be supplied. Defaulting to 50');
    end

    if strcmpi(obj.Method,'SGD')
        if ~isfield(obj, 'LearningRate')|| ~isfield(obj, 'MiniBatchSize')
            error('At least learning rate and mini batch size must be supplied for standard gradient descent!');
        end

        if isvector(obj.LearningRate)
            if numel(obj.LearningRate) ~= obj.Iter
                error('Invalid learning rate.');
            end
            obj.LearningAnnealed = 'yes';
        else
            obj.LearningAnnealed = 'no';
        end
    elseif strcmpi(obj.Method,'Momentum')
        if ~isfield(obj, 'LearningRate')|| ~isfield(obj, 'MiniBatchSize') || ~isfield(obj, 'Momentum')
            error('Learning rate , momentum and mini batch size must be supplied for SGD with momentum!');
        end
        
        if ~isscalar(obj.LearningRate) && isvector(obj.LearningRate)
            if numel(obj.LearningRate) ~= obj.Iter
                error('Invalid learning rate.');
            end
            obj.LearningAnnealed = 'yes';
        else
            obj.LearningAnnealed = 'no';
        end
        if ~isscalar(obj.Momentum) && isvector(obj.Momentum)
            if numel(obj.Momentum) ~= obj.Iter
                error('Invalid momentum.');
            end            
            obj.MomentumAnnealed = 'yes';
        else
            obj.MomentumAnnealed = 'no';
        end
    elseif strcmpi(obj.Method,'NAG')
        if ~isfield(obj, 'LearningRate')|| ~isfield(obj, 'MiniBatchSize') || ~isfield(obj, 'Momentum')
            error('Learning rate , momentum and mini batch size must be supplied for SGD with nesterov accelerated gradient!');
        end
        
        if ~isscalar(obj.LearningRate) && isvector(obj.LearningRate)
            if numel(obj.LearningRate) ~= obj.Iter
                error('Invalid learning rate.');
            end
            obj.LearningAnnealed = 'yes';
        else
            obj.LearningAnnealed = 'no';
        end
        if ~isscalar(obj.Momentum) && isvector(obj.Momentum)
            if numel(obj.Momentum) ~= obj.Iter
                error('Invalid momentum.');
            end            
            obj.MomentumAnnealed = 'yes';
        else
            obj.MomentumAnnealed = 'no';
        end
    elseif strcmpi(obj.Method,'ADADelta')            
        if ~isfield(obj, 'MiniBatchSize') || ~isfield(obj, 'Decay') || ~isfield(obj, 'e')
            error('Mini batch size, Delta decay and constant E must be supplied for ADADelta.');
        end
    elseif strcmpi(obj.Method,'Decay')
        if ~isfield(obj, 'MiniBatchSize') || ~isfield(obj, 'MaxNorm') || ~isfield(obj, 'Saturation')
            error('At mini batch size, MaxNorm constraints and Saturation epoch must be set for gradient descent with decay!');
        end
        if ~isfield(obj, 'LearningRate')
            obj.LearningRate = 10;
        end    
        if ~isfield(obj, 'Momentum')
            obj.Momentum = 0.5;
        end
        if obj.Iter < obj.Saturation
            warning('Number of epochs is lower than the saturation point.');
        end
    else
        error('Gradient descent method not selected or unrecognized.')
    end

end