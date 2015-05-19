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

function [J, Grad] = ffnn_Regularizer(nn, options, X, C)

    [J, grad] = ffnn_loss(nn, X, C); %Gradient Descent branch

    if ~strcmpi(options.Regularizer,'none')
        lambda = options.RegLambda;
        error('Inconsistente');
        reg = ['penalty',options.Regularizer];
        theta_cum = cumsum([0,(1+topology(1,1:(end-1))).*topology(1,2:end)]);

        Penalty = 0;
        Grad = zeros(size(nn_params));
        argstr = 'feval(reg, t)';

        for layer=1:(length(topology)-1);
            t = decompress(nn_params, topology, layer);
            [LocalPenalty, LocalGrad] = eval(argstr);

            Penalty = Penalty + LocalPenalty;

            Grad(theta_cum(layer)+1:theta_cum(layer+1)) = ...
                decompress(grad,topology,layer) + lambda/m*LocalGrad;
        end 
        J = J + lambda/(2*m)*Penalty;        
    else
        Grad = grad;
    end
end

%Calculate additive penalties for lasso.

%We compute lasso with prof. Rui Rodrigues suggestion of using
%the abs(theta) approximation: sqrt(theta.^2 + epsilon).

function [J, Grad] = penaltylasso(theta)

    epsilon = 0.0001;
    %Auxiliar variables
    t = [zeros(1,size(theta,2));theta(2:end,:)];            %Never regularize bias terms
    tmp = sqrt(t.^2 + epsilon);
    
    J = sum(sum(tmp,1),2);    
    Grad = t./(tmp);
end

%Calculate additive penalties for tikhonov.

function [J, Grad] = penaltytikhonov(theta)
    J = sum(sum(theta(2:end,:).^2,1),2);                    %Never regularize bias terms
    Grad = [zeros(1,size(theta,2));theta(2:end,:)];
end