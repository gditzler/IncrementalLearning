function [net,f_measure,g_mean,recall,precision,err] = learn_nie(net, data_train, labels_train, data_test, labels_test)
% implement learn++.nie

disp('I just wrote this up. I have not fully tested it. So')
disp('use it at your own risk. Let me know of you catch any')
disp('errors.      -Gregory')
disp(' ')
disp('Use open a matlabpool for parallel processing. Should')
disp('speed up the runtime quite a bit. ')


n_timestamps = length(data_train);  % total number of time stamps
net.classifiers = {};   % classifiers
net.w = [];             % weights 
net.initialized = false;% set to false
net.t = 1;              % track the time of learning
net.classifierweigths = {};      % array of classifier weights
net.type = 'learn++.nie';

f_measure = zeros(n_timestamps, net.mclass);
g_mean = zeros(n_timestamps, 1);
recall = zeros(n_timestamps, net.mclass);
precision = zeros(n_timestamps, net.mclass);
err = zeros(n_timestamps, 1);


for ell = 1:n_timestamps
  
  % get the training data for the 't'th round 
  data_train_t = data_train{ell};
  labels_train_t = labels_train{ell};
  data_test_t = data_test{ell};
  labels_test_t = labels_test{ell};
    
  % has the ensemble been initialized
  if net.initialized == false,
    net.beta = [];
  end
  
  % STEP 1:
  net.classifiers{end + 1} = bagging_variation(...
    data_train_t, ...
    labels_train_t, ...
    net.n_classifiers, ...
    net.minority_class, ...
    net.base_classifier);
  
  % STEP 2: Evaluate all existing classifiers on new data
  t = size(net.classifiers,2);
  y = decision_ensemble(net, data_train_t, labels_train_t, t);

  for k = 1:net.t
    
    epsilon_tk = return_espilon_kt(net, labels_train_t, y(:, k));
    if k == net.t
      disp([epsilon_tk,net.t])
    end
    
    if (k<net.t)&&(epsilon_tk>0.5) 
      epsilon_tk = 0.5;
    elseif (k==net.t)&&(epsilon_tk>0.5)
      % try generate a new classifier 
      net.classifiers{k} = bagging_variation(...
        data_train_t, ...
        labels_train_t, ...
        net.n_classifiers, ...
        net.minority_class, ...
        net.base_classifier);
      epsilon_tk = return_espilon_kt(net, ...
        labels_train_t, ...
        subensemble_test(net.classifiers{k}, ...
          data_train_t, ...
          net.mclass) ...
        );
      epsilon_tk(epsilon_tk > 0.5) = 0.5;
    end
    net.beta(net.t,k) = epsilon_tk / (1-epsilon_tk);
  end
  
  % compute the classifier weights
  if net.t==1,
    if net.beta(net.t,net.t)<net.threshold,
      net.beta(net.t,net.t) = net.threshold;
    end
    net.w(net.t,net.t) = log(1/net.beta(net.t,net.t));
  else
    for k = 1:net.t,
      b = t - k - net.b;
      omega = 1:(net.t - k + 1); 
      omega = 1./(1+exp(-net.a*(omega-b)));
      omega = (omega/sum(omega))';
      beta_hat = sum(omega.*(net.beta(k:net.t,k)));
      if beta_hat < net.threshold,
        beta_hat = net.threshold;
      end
      net.w(net.t,k) = log(1/beta_hat);
    end
  end
  
  % STEP 7: classifier voting weights
  net.classifierweigths{end+1} = net.w(end,:);
  
  
  [predictions,posterior] = classify_ensemble(net, data_test_t, labels_test_t);
  [f_measure(ell,:),g_mean(ell),recall(ell,:),precision(ell,:),...
    err(ell)] = stats(labels_test_t, predictions, net.mclass);

  net.initialized = 1;
  net.t = net.t + 1;

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUXILARY FUNCTIONS
function y = decision_ensemble(net, data, labels, t)
y = zeros(numel(labels), t);
for k = 1:t
  y(:, k) = subensemble_test(net.classifiers{k}, data, net.mclass);
end


function predictions = subensemble_test(classifiers, data, mclass)
n_experts = length(classifiers);        % how many classifiers 
weights = ones(n_experts, 1)/n_experts; % uniform weights
p = zeros(size(data,1), mclass);
y = zeros(size(data,1), n_experts);

parfor k = 1:n_experts
  y(:,k) = classifier_test(classifiers{k}, data);  
end

for k = 1:n_experts
  for m = 1:numel(y)
    p(m,y(m,k)) = p(m,y(m,k)) + weights(k);
  end
end
[~,predictions] = max(p');
predictions = predictions';


function [predictions,posterior] = classify_ensemble(net, data, labels)
n_experts = length(net.classifiers);
weights = net.w(end,:);
if n_experts ~= length(weights)
  error('What are there are different number of weights and experts!')
end
p = zeros(numel(labels), net.mclass);
for k = 1:n_experts
  y = subensemble_test(net.classifiers{k}, data, net.mclass);
  
  % this is inefficient, but it does the job 
  for m = 1:numel(y)
    p(m,y(m)) = p(m,y(m)) + weights(k);
  end
end
[~,predictions] = max(p');
predictions = predictions';
posterior = p./repmat(sum(p,2),1,net.mclass);


function epsilon_tk = return_espilon_kt(net, labels, y)
if strcmp(net.method, 'wavg')
  [~,~,recl] = stats(labels, y, net.mclass);
  epsilon_tk = sum(net.cvx_parameter.*(1 - recl));
elseif strcmp(net.method, 'fm')
  fm = stats(labels, y, net.mclass);
  epsilon_tk = 1 - fm(net.minority_class);
elseif strcmp(net.method, 'gm')
  [~,gm] = stats(labels, y, net.mclass);
  epsilon_tk = 1 - gm;
else
  error('LEARN_NIE :: Unknown weighting method!')
end
