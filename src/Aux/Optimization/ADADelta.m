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


%% AdaDelta, Matthew D. Zeiler in "ADADELTA: An adaptive learning rate method"
function [J, Update, accum_grad, accum_updt] = ADADelta(nn, options, X, Y, accum_grad, accum_updt)

    [J, grad] = ffnn_Regularizer(nn, options, X, Y);

    accum_grad = options.Decay*accum_grad + (1-options.Decay)*(grad.^2);
    Update = compute_update(accum_updt, accum_grad).*grad; %This scales the update to the same unit as the nn parameters.
    accum_updt = options.Decay*accum_updt + (1-options.Decay)*(Update.^2);

    function out = compute_update(accum_grad, accum_updt)
        out = sqrt((accum_updt + options.e)./(accum_grad + options.e));
    end
end