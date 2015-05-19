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

function X = tanh_sigmoid_fast(x)

%Lambert's continued fraction
%https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
%Changes made, Tanh adapted for faster learning, see le cun.

x = 0.666667*x;
x2 = x.^2;
a = 1.7159 * x .* (135135 + x2 .* (17325 + x2 .* (378 + x2)));
b = 135135 + x2 .* (62370 + x2 .* (3150 + x2 .* 28));
X = a./b;
end