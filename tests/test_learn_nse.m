function test_learn_nse()
% test learn++.nse

disp('The ConceptDriftData.m file must be in the Matlab path. This');
disp('file can be found: https://github.com/gditzler/ConceptDriftData ');
addpath('../src/');

model.type = 'CART';
net.a = .5;
net.b = 10;
net.threshold = 0.01; 
net.mclass = 2;
net.base_classifier = model;

T = 200;
N = 100;
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('sea', T, N);
for t = 1:T
  data_train{t} = data_train{t}';
  labels_train{t} = labels_train{t}';
  data_test{t} = data_test{t}';
  labels_test{t} = labels_test{t}';
end

[errs_nse, net_nse] = learn_nse(net, data_train, labels_train, data_test, ...
  labels_test);

smote_params.minority_class = 2;
smote_params.k = 3;
smote_params.N = 200;
[errs_cds, net_cds] = learn_nse(net, data_train, labels_train, data_test, ...
  labels_test, smote_params);

figure;
plot(errs_nse)
plot(errs_cds,'r')

net_nse
net_cds
