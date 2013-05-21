function [net,errs] = learn(net, data_train, labels_train, data_test, labels_test)
% learn++

%     learn.m
%     Copyright (C) 2013 Gregory Ditzler
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.


Tk = net.iterations;
K = length(data_train);
net.classifiers = cell(Tk*K, 1); 
net.beta = zeros(Tk*K, 1); 

c_count = 0;
errs = zeros(Tk*K, 1);


for k = 1:K  
  data_train_k = data_train{k};
  labels_train_k = labels_train{k};
  D = ones(numel(labels_train_k), 1)/numel(labels_train_k);
  
  if k > 1
    predictions = classify_ensemble(net, data_train_k, labels_train_k, ...
      c_count);
    epsilon_kt = sum(D(predictions ~= labels_train_k));
    beta_kt = epsilon_kt/(1-epsilon_kt);
    D(predictions == labels_train_k) = beta_kt * D(predictions == labels_train_k);
  end
  
  for t = 1:Tk
    c_count = c_count + 1;
    
    % step 1
    D = D / sum(D);
    
    % step 2
    index = randsample(1:numel(D), numel(D), true, D);
    
    % step 3
    net.classifiers{c_count} = classifier_train(...
      net.base_classifier, ...
      data_train_k(index, :), ...
      labels_train_k(index));
    
    % step 4
    y = classifier_test(net.classifiers{c_count}, data_train_k);
    epsilon_kt = sum(D(y ~= labels_train_k));
    net.beta(c_count) = epsilon_kt/(1-epsilon_kt);
    
    % step 5
    predictions = classify_ensemble(net, data_train_k, labels_train_k, ...
      c_count);
    E_kt = sum(D(predictions ~= labels_train_k));
    if E_kt > 0.5
      % rather than remove remove existing classifier; null the result out
      % by forcing the loss to be equal to 1/2 which is the worst possible
      % loss. feel free to modify the code to go back an iteration. 
      E_kt = 0.5;   
    end
    
    % step 6
    Bkt = E_kt / (1 - E_kt);
    D(predictions == labels_train_k) = Bkt * D(predictions == labels_train_k);
    D = D / sum(D);
    
    % make some predictions 
    [predictions,posterior] = classify_ensemble(net, data_test, ...
      labels_test, c_count);
    errs(c_count) = sum(predictions ~= labels_test)/numel(labels_test); 
  end
  
  
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUXILARY FUNCTIONS
function [predictions,posterior] = classify_ensemble(net, data, labels, lims)
n_experts = lims;
weights = log(1./net.beta(1:lims));
p = zeros(numel(labels), net.mclass);
for k = 1:n_experts
  y = classifier_test(net.classifiers{k}, data);
  
  % this is inefficient, but it does the job 
  for m = 1:numel(y)
    p(m,y(m)) = p(m,y(m)) + weights(k);
  end
end
[~,predictions] = max(p');
predictions = predictions';
posterior = p./repmat(sum(p,2),1,net.mclass);



