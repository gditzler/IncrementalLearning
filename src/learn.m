function [net,errs] = learn(net, data_train, labels_train, data_test, labels_test)
% learn++

net.base_classifier = 'CART';
net.iterations = 10;

Tk = net.iterations;
K = length(data_train);
net.classifiers = cell(Tk*K, 1); 
net.beta = zeros(Tk*K, 1); 

c_count = 1;
errs = zeros(Tk*K, 1);


for k = 1:K
  data_train_k = data_train{k};
  labels_train_k = labels_train{k};
  
  if k == 1
    D = ones(numel(labels_train_k), 1)/numel(labels_train_k);
  else
    predictions = classify_ensemble(net, data_train_k, labels_train_k, ...
      c_count);
    epsilon_kt = sum(D(predictions ~= labels_train_k));
    beta_kt = epsilon_kt/(1-epsilon_kt);
    D(predictions == labels_train_k) = beta_kt * D(predictions == labels_train_k);
  end
  
  for t = 1:Tk
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
    
    c_count = c_count + 1; 
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



