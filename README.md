# Project_Cifar-10
使用不同的优化器RMSprop、Adagrad、SGD和Adam优化器对Cifar-10数据集进行实验，使用Accuracy、Precision、Recall、F1-score四个评价指标；
对于每个优化器根据实验结果进行迭代次数、学习率的调整以提高准确率，但随着学习率的调整，准确率提高的变化幅度并不是很大，都不超过70%；
最后，进行了模型框架的调整，重新构建了一个卷积网络，使用SGD优化器，lr=0.01时，训练后的准确率达到了86%。
