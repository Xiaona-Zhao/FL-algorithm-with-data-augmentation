
# Federated Learning with Generative Adversarial Network to Improve Training on Non-i.i.d Data

This repository shows the pytorch code for Dissertation Report in applying MSc in Information Technology, The Hong Kong Polytechnic University.


PDF.file will be uploaded after defence.

Welcome to make contact with author: xiaona.zhao@connect.polyu.hk :smiley:


## Abstract 
This work will focus on solving the impact of statistical heterogeneity on data distribution between clients in Federated Learning. 

By generating simulated data using the Adversarial Generation Network (GAN), defects of statistical heterogeneity between clients should be smoothed in the distributed model, thereby improve the test accuracy as well as the convergence rate in distributed learning framework. 

More specifically, compared to the current improving methods such as distributed optimization method[8-16], client clustering method[17-22] and the combination between reinforcement learning and federated learning[35-38] that have been wildly used in journals published in the last two years, this work attempt to use an innovative method to generate synthetic data by deploying Multi_Discriminator GAN on the server side. 

Considering privacy guarantee of federated learning, the discriminators will be trained based on local dataset so that raw data exchange is prohibited between clients and server. Non_iid assumption is the major reason of the data distribution drift between clients, and a multi_path generator is implemented on server side to cover multimodel distribution of discriminators in GAN training. 

To perform a quantitative analysis on the influence of non_iid data have on model performance in federated learning algorithm, Dirichlet distribution function is implemented in data preprocess step with a tunable parameter non_iid_alpha in chapter 2. Experiments on synthesize dataset shows ideal performance of multi_path generator on multimodel distributed training dataset, and a visualization of the generated images from multi_path generator GAN in non_iid dataset is shown in chapter 3. Chapter 4 provide the global model performance evaluation, showing the improvement of classification accuracy on full batch gradient descent algorithm from 0.54 to 0.90, as well as in FedAvg algorithm which test accuracy increase from 0.48 to 0.86. A convergence analysis on local training epoch of a specific client is also given, which can be another insight evidence on the performance improvement of GAN based data augmentation to federated learning under non_iid assumption.

