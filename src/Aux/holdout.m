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

function [Xfirst, yfirst, Xsecond, ysecond] = holdout(X,y, prop)

    [m, p] = size(X);
    
    class = unique(y);
    cs = length(class);
    
    %Sort/Organize by class's    
    [y, idx] = sortrows(y);
    X = X(idx,:);

    Dist = accumarray(y,1);

    fprintf('\nOriginal distributions:\n\n');
    for i=1:cs
        fprintf('Class %4i | Instances: %8d | Proportion: %.4f\r', class(i), Dist(i), Dist(i)/sum(Dist));        
    end
    
    %TODO, make it more robust.
    firstSize = ceil(m*prop);
    if mod(firstSize,length(class)) ~= 0
        warning('Resulting dataset is not distributed evenly.');
    end

   
    Xfirst = zeros(firstSize, p);
    yfirst = zeros(firstSize, 1);
    Xsecond = zeros(m-firstSize, p);
    ysecond = zeros(m-firstSize, 1);

    Xtr = repmat(floor(firstSize/cs),cs,1);
    Xte = Dist-Xtr;
    
    X_idx = [0;cumsum(Dist)];
    Xtr_idx = [0;cumsum(Xtr)];
    Xte_idx = [0;cumsum(Xte)];
    
    %Distribute class's
    for i=1:cs
        Xfirst((Xtr_idx(i)+1):Xtr_idx(i+1),:)  = X((X_idx(i)+1):(X_idx(i)+Xtr(i)),:);
        yfirst((Xtr_idx(i)+1):Xtr_idx(i+1),:)  = y((X_idx(i)+1):(X_idx(i)+Xtr(i)),:);
        Xsecond((Xte_idx(i)+1):Xte_idx(i+1),:) = X((X_idx(i)+Xtr(i)+1):(X_idx(i+1)),:);
        ysecond((Xte_idx(i)+1):Xte_idx(i+1),:) = y((X_idx(i)+Xtr(i)+1):(X_idx(i+1)),:);
    end
    
    trainDist = accumarray(yfirst,1);
    testDist = accumarray(ysecond,1);
    
    fprintf('\nStratified Data:\n\n%33s\t%15s\n','First','Second');
    for i=1:cs
        fprintf('Class %4i | Instances: %8i | Instances: %8i\r',i,trainDist(i),testDist(i));
    end
end