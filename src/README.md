IncrementalLearning
===================

I am planning on providing some implementations of the Learn++ algorithm in Matlab. Please direct any questions, comments, or suggestions to <gregory.ditzler@gmail.com>. More details to come.



About the functions 
===================
* `classifier_train.m` - implementation of a generic build of a classifier. Refer to the documentation on how to call this function, but in summary you pass it a structure indicating some information about the classifier you would like to create (e.g., CART with some pruning options), and the training data + labels. The result is a new structure containing information about the classifier and the classifier is saved as one of the fields. 
* `classifier_test.m` - test a classifier that was trained using classifier_train.m. Refer to the documentation on how to call this function.
* `learn.m` - implementation of the original Learn++ algorithm. The base classifier is controlled using `classifier_train.m` and `classifier_test.m`. Refer to the documentation on how to call this function.
* `learn_nse.m` - implementation of the original Learn++ algorithm. The base classifier is controlled using `classifier_train.m` and `classifier_test.m`. Refer to the documentation on how to call this function.
* `smote.m` - synthetic minority oversampling technique. Refer to the documentation on how to call this function.


References 
===================
1. R. Polikar, L. Udpa, S. Udpa, and V. Honavar, "Learn++: An incremental learning algorithm for supervised neural networks," IEEE Transactions on System, Man and Cybernetics (C), Special Issue on Knowledge Management, vol. 31, no. 4, pp. 497-508, 2001.
2. R. Elwell and R. Polikar, "Incremental Learning of Concept Drift in Nonstationary Environments" IEEE Transactions on Neural Networks, vol. 22, no. 10, pp. 1517-1531
3. G. Ditzler and R. Polikar, "Incremental learning of concept drift from streaming imbalanced data," in IEEE Transactions on Knowledge & Data Engineering, 2012, accepted.
4. N. V. Chawla, K. W. Bowyer, T. E. Moore and P. Kegelmeyer, "SMOTE: Synthetic Minority Over-Sampling Technique," Journal of Artificial Intelligence Research, 16, 321-357, 2002.
