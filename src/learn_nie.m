function learn_nie(net, data_train, labels_train, data_test, labels_test)
% implement learn++.nie
net.method = 'wavg';% wavg, fm, gm


disp('I just wrote this up. I have not fully tested it. So')
disp('use it at your own risk. Let me know of you catch any')
disp('errors.      -Gregory')


n_timestamps = length(data_train);  % total number of time stamps
net.classifiers = {};   % classifiers
net.w = [];             % weights 
net.initialized = false;% set to false
net.t = 1;              % track the time of learning
net.classifierweigths = {};      % array of classifier weights
errs = zeros(n_timestamps, 1);


