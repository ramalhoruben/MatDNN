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


function C = softmax(c)

    %I substract the max C (over each obs) from the remaining as to avoid 
    %overflows and numerical problems.
    tmp = exp(bsxfun(@minus, c, max(c, [], 1)));
    C = bsxfun(@rdivide, tmp, sum(tmp,1));
    %Log of exact 0 will generate exactly -infinity (carefull)
    if sum(sum(isnan(C)))>0
        C
        tmp
        c
        pause
    end
    C(C == 0) = realmin;
end