function learn(net, data_train, labels_train, data_test, labels_test)
% learn++

net.base_classifier = 'CART';
net.iterations = 10;

Tk = net.iterations;
K = length(data_train);
net.classifiers = cell(Tk*K, 1); 
net.beta = zeros(Tk*K, 1); 

c_count = 1;


for k = 1:K
  data_train_k = data_train{k};
  labels_train_k = labels_train{k};
  
  if k == 1
    D = ones(numel(labels_train_k), 1)/numel(labels_train_k);
  else
  end
  
  for t = 1:Tk
    D = D / sum(D);
    index = randsample(1:numel(D), numel(D), true, D);
    net.classifiers{c_count} = classifier_train(...
      net.base_classifier, ...
      data_train_k(index, :), ...
      labels_train_k(index));
    
    y = classifier_test(net.classifiers{c_count}, data_train_k);
    epsilon_kt = sum(D(y ~= labels_train_k));
    net.beta(c_count) = epsilon_kt/(1-epsilon_kt);
  end
  
  
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUXILARY FUNCTIONS
function [predictions,posterior] = classify_ensemble(net, data, labels, lims)
n_experts = length(net.classifiers);
weights = net.w(end,:);
if n_experts ~= length(weights)
  error('What are there are different number of weights and experts!')
end
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



