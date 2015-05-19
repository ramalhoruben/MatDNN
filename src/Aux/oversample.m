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


function [XOut, yOut] = oversample(X, y, sizePerClass, print)
    
    [~, p] = size(X);
    class = unique(y);
    cs = length(class);
    
    XOut = zeros(sizePerClass*cs,p);
    yOut = zeros(sizePerClass*cs,1);

    
    %Sort/Organize by class's    
    [y, idx] = sortrows(y);
    X = X(idx,:);

    Dist = accumarray(y,1);

    if strcmpi(print,'on') || strcmpi(print,'yes')
        fprintf('\nOriginal distributions:\n\n');
        for curClass=1:cs
            fprintf('Class %4i | Instances: %8d | Proportion: %.4f\r', ...
                class(curClass), Dist(curClass), Dist(curClass)/sum(Dist));
            if Dist(curClass)>sizePerClass
                error('SizePerClass is too low.')
            end
        end
    end
    for curClass=1:cs
        classData = [X(y==curClass,:),y(y==curClass)];

        oversampledClassData = datasample(classData,sizePerClass - Dist(curClass),1);
        XOut(sizePerClass*(curClass-1)+1:sizePerClass*curClass,:) = [classData(:,1:end-1);oversampledClassData(:,1:end-1)];
        yOut(sizePerClass*(curClass-1)+1:sizePerClass*curClass) = [classData(:,end);oversampledClassData(:,end)];
    end

end