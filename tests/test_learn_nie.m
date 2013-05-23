% function test_learn_nie()
clc
clear
close all

disp('The ConceptDriftData.m file must be in the Matlab path. This');
disp('file can be found: https://github.com/gditzler/ConceptDriftData ');
addpath('../src/');



model.type = 'CART';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct
net.minority_class = 2;
net.cvx_parameter = .5*ones(1,2);
net.method = 'fm';% wavg, fm, gm
net.n_classifiers = 5;
net.minority_class = 2;



% generate the sea data set
T = 200;  % number of time stamps
N = 100;  % number of data points at each time
ds = .5;
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('sea', T, N);
for t = 1:T
  % i wrote the code along time ago and i used at assume column vectors for
  % data and i wrote all the code for learn++ on github to assume row
  % vectors. the primary reasoning for this is that the stats toolbox in
  % matlab uses row vectors for operations like mean, cov and the
  % classifiers like CART and NB
  d = data_train{t}';
  l = labels_train{t}';
  ind = find(l == net.minority_class);
  ind = ind(randperm(numel(ind)));
  ind = ind(1:floor(numel(ind)*ds));
  d(ind,:) = [];
  l(ind) = [];
  data_train{t} = d;
  labels_train{t} = l;

  d = data_test{t}';
  l = labels_test{t}';
  ind = find(l == net.minority_class);
  ind = ind(randperm(numel(ind)));
  ind = ind(1:floor(numel(ind)*ds));
  d(ind,:) = [];
  l(ind) = [];
  data_test{t} = d;
  labels_test{t} = l;
end

[net,f_measure,g_mean,recall,precision,err] = learn_nie(net, ...
  data_train, labels_train, data_test, labels_test);
