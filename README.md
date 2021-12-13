## Federated Learning with Generative Adversarial Network to Improve Training on Non-i.i.d Data.


This is the code accompanying the Dissertation Project.  
The dissertation report is for application of M.Sc. in Information Technology, The Hong Kong Polytechnic University.

#### Requirements to run the code:
---

1. Python 3.7
2. PyTorch 1.0.2
3. Scikit-learn
4. Matplotlib
5. Numpy


#### Important source files:
---

1. `FedAVG.py`: Implement of the original Federated Average algorithm, which is used as the baseline.
2. `Federated_Learning_with_Aug.py`: Implement of the FL algorithm with GAN based data augmentation, which is the main function of the project.


#### Important arguments:
---


The following arguments to the FedAvg_training.py file control the important parameters of the experiment

1. `FL_epochs`: Defines the number of federated learning training epoch.
2. `Gan_epochs`: Defines the number of Multi_GAN training epoch.
3. `Generator_paths`: Number of the path in Generator of the GAN, set to be '4' as a better performance.
4. `non_iid_alpha`: Set for data partition. Set to be 100 for iid data, and 0.01 for non_iid data, quantitative is also allowed.
5. `iid`: Set for original partition method 'True' of 'False', when data to be partitioned based on the same way in Federated Average algorithm.


#### Output:
---

The training_loss and test_accuracy is the main output, that can evaluate the model performance.  
Also, the majority of the information is logged to a log file in the log folder.


### Make Contact:
---

```
xiaona.zhao@connect.polyu.hk
```
