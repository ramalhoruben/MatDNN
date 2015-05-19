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

function [bst_nn,JTrain, JValid] = gradientDescent(nn, options, X, C)

    JValid = [];
    % Read options
    if exist('options', 'var') && ~isempty(options)
        if options.MiniBatchSize > size(X,2)
            error('The mini batch size is higher than the number of observations!');
        end
        
        mini_batch_size = options.MiniBatchSize;        
        it = options.Iter;
        if strcmpi(options.Method,'ADADelta')
            update_method = 'ADADelta(nn, options, mini_batch_X{b}, mini_batch_C{b}, arg1, arg2)';
        elseif strcmpi(options.Method,'SGD')
            update_method = 'SGD(nn, options, mini_batch_X{b}, mini_batch_C{b}, epoch)';            
        elseif strcmpi(options.Method,'Momentum')
            update_method = 'Momentum(nn, options, mini_batch_X{b}, mini_batch_C{b}, arg1, epoch)';
        elseif strcmpi(options.Method,'NAG')
            %NAG version of momentum
            update_method = 'Momentum(nn, options, mini_batch_X{b}, mini_batch_C{b}, arg1, epoch)';
        elseif strcmpi(options.Method,'Decay')
            init_newth = options.LearningRate;
            init_momentum = options.Momentum;
            final_momentum = 0.99;
            saturation = options.Saturation;
            maxnorm = options.MaxNorm;
            update_method = 'Decay(grad)';            
        end
        
        maxnorm = options.MaxNorm;

    else
        error('Must supply options.');
    end
    
    nn_params = unroll(nn.layer_weights, nn);        
    
    if isfield(options, 'validSet')
        Xvalid = options.validSet(:,1:(end-1));
        cvalid = options.validSet(:,end);
        Cvalid = onehot(cvalid);
        JValid = zeros(it,1);
    end
    
    [n,m] = size(X);
    csize = size(C,1);
    
    %Mini-Batch size might not be an exact divisor for the dataset size, 
    %correction for this issue:
    if mod(m,mini_batch_size) == 0
        num_batchs = m/mini_batch_size;
        batch_sizes = repmat(mini_batch_size,1,num_batchs);
    else
        num_batchs = ceil(m/mini_batch_size);
        batch_sizes = [repmat(mini_batch_size,1,num_batchs-1), mod(m, mini_batch_size)];
    end
        
    %Some usefull variables
    
    arg1 = 0;
    arg2 = 0;
    last_grad = 0;
    bst_J = Inf;
    JTrain = zeros(it,1);
    for epoch=1:it
        shuffled = randperm(m);
        
        % Shuffle and split in batchs.
        mini_batch_X = mat2cell(X(:,shuffled),n,batch_sizes);
        mini_batch_C = mat2cell(C(:,shuffled),csize,batch_sizes);
        
        accumJ = 0;        
        for b=1:num_batchs
            [J, update, arg1, arg2] = eval(update_method);

            %Accumulate the sum of the loss's over all mini batchs
            accumJ = accumJ + J*mini_batch_size;
            
            %Update weights            
            nn_params = maxNorm(nn_params - update, nn, maxnorm);
            
            %Update network
            nn.layer_weights = updateNet(nn_params, nn, options.Method);
         end
       
        JTrain(epoch,1) = accumJ/num_batchs;    %Calculate the mean loss cost

        if ~isfield(options, 'validSet')
            fprintf('Epoch %4i | Loss: %4.6e\r', epoch, JTrain(epoch,1)); 
            if JTrain(epoch,1) < bst_J
                bst_J = JTrain(epoch,1);
                bst_nn = nn;
            end
        else
            [~, Chat_valid] = predict_ffnn(nn, Xvalid);
            JValid(epoch,1) = nn.cost_function(Chat_valid, Cvalid);            
            fprintf('Epoch %4i | Loss: %4.6e, Valid: %4.6e\r', epoch, JTrain(epoch,1), JValid(epoch,1)); 
            if JValid(epoch,1) < bst_J
                bst_J = JValid(epoch,1);
                bst_nn = nn;
            end
        end
    end
    if ~isfield(options, 'validSet')
        plot(1:epoch,JTrain);
        legend('Train Error');
    else
        plot(1:epoch,JTrain,1:epoch,JValid);
        legend('Train Error','Valid Error');
    end
    xlabel('Epoch');
    ylabel('Error');
          
    function update = Decay(grad)
        progress = epoch/saturation;
        dynamicMomentum = min(progress*init_momentum + (1-progress)*final_momentum, 0.99);
        dynamicNewth = init_newth*0.998^epoch;
        last_grad = dynamicMomentum * last_grad - (1-dynamicMomentum)*dynamicNewth*maxNorm(nn_params - grad, nn, maxnorm);
        update = -last_grad;
    end

    function unrolled = unroll(rolled, nn)
       unrolled = zeros(nn.total_weights,1);
       for layer=1:nn.tot_layers
           unrolled((nn.layer_idxs(layer)+1):nn.layer_idxs(layer+1),1) = rolled{layer};
       end
    end

    function layer_weights = updateNet(unrolled, nn, method)
        if strcmpi(method,'NAG')
            if isscalar(options.Momentum)
                unrolled = unrolled + options.Momentum.*arg1;                
            elseif isvector(options.Momentum)
                unrolled = unrolled + options.Momentum(epoch).*arg1;
            end
        end
        layer_weights = cell(nn.tot_layers,1);
        for layer=1:nn.tot_layers
            layer_weights{layer} = unrolled((nn.layer_idxs(layer)+1):(nn.layer_idxs(layer+1)),1);
        end
    end
end