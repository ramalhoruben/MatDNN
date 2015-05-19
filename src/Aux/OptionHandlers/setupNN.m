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


%% Available Options:
%  Network topology   : [input, hidden1, hidden2, ..., hiddenN , output]
%  Activation Function:  ReLu    (Default) | Tanh     | InvLog)
%  Output Function    :  Softmax (Default) | InvLog
%  Cost Function      :  Neg. Log Likelihood (Default)

function nn = setupNN(varargin)

    allfields = {'Topology';'Activation';'ActivationOutput'; ...
        'CostFunction';'Dropout';'CostSensitive'};

    for i=1:nargin
        arg = varargin{i};
        for j=1:size(allfields)
           if strcmpi(arg,allfields{j})
               inArg.(allfields{j}) = varargin{i+1};
               i = i+1;
           end
        end
    end
    if ~isfield(inArg, 'Topology')
        error('Topology is not optional!');
    end
    %Optional fields
    if ~isfield(inArg, 'Activation')
        inArg.Activation = 'relu';
    end
    if ~isfield(inArg,'ActivationOutput')
        inArg.ActivationOutput = 'softmax'; 
    end
    if ~isfield(inArg,'CostFunction')
        inArg.CostFunction = 'negloglikelihood';
    end
    
    nn.topology = inArg.Topology;                                          %Network topology
    nn.tot_layers = length(nn.topology)-1;                                 %Total number of hidden layers
    nn.layer_size = (1+nn.topology(1,1:(end-1))).*nn.topology(1,2:end);    %Total number of weights per layer
    nn.total_weights = sum(nn.layer_size);                                 %Total number of weights in the whole net
    nn.layer_idxs = cumsum([0,nn.layer_size]);                             %Weight, per layer indexs
    nn.num_classes = nn.topology(1,end);

    if ~isfield(inArg,'Dropout')
        nn.dropout = 'no';
    else
        nn.dropout = 'yes';
        nn.dropout_p = inArg.Dropout;
    end
    
    if ~isfield(inArg,'CostSensitive')
        nn.CostSensitive = 'no';
    else
        proportions = inArg.CostSensitive';        
        nn.CostSensitive = 'yes';
        nn.CostWeights = (1/sum(1./proportions))./proportions;
    end

    
    if strcmpi(inArg.Activation,'relu')
        nn.weight_init = @(a,b,c)(relu_initWeights(a,b,c));
        nn.activation_function = @(x)(relu(x));       
        nn.activation_derivative = @(x)(relu_grad(x));
    elseif strcmpi(inArg.Activation,'tanh')
        nn.weight_init = @(a,b,c)(tanh_initWeights(a,b,c));
        nn.activation_function = @(x)(tanh_sigmoid(x));
        nn.activation_derivative = @(x)(tanh_sigmoid_grad(x));
    elseif strcmpi(inArg.Activation,'invlog')
        nn.weight_init = @(a,b,c)(log_initWeights(a,b,c));
        nn.activation_function = @(x)(log_sigmoid(x));
        nn.activation_derivative = @(x)(log_sigmoid_grad(x));
    else
        error(['Unrecognized activation function: "',inArg.Activation,'"']);
    end
    
    if strcmpi(inArg.ActivationOutput,'softmax')
        nn.activation_function_output = @(x)(softmax(x));
    elseif strcmpi(inArg.ActivationOutput,'InvLog')
        nn.activation_function_output = @(x)(log_sigmoid(x));
    else
        error(['Unrecognized output function: "',inArg.ActivationOutput,'"']);
    end
    
    if strcmpi(inArg.CostFunction,'negLogLikelihood')
        nn.cost_function = @(C_hat, C)(negLogLikelihood(C_hat, C));
    elseif strcmpi(inArg.CostFunction,'crossentropy')
        nn.cost_function = @(C_hat, C)(crossEntropy(C_hat, C));
    else
        error(['Unrecognized cost function: "',inArg.CostFunction,'"']);        
    end
    
    nn.layer_weights = initWeights(nn.topology, nn.layer_size, nn.tot_layers, nn.weight_init);
end
