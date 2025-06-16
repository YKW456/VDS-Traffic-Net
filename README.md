# VDS-Traffic-Net
Used for classifying Vt values and traffic images

1. Project Overview
   
This project focuses on deep learning tasks related to traffic scenarios, involving network model construction, dataset creation and processing, as well as training and visualization processes for different tasks. Implement the complete process from data preparation to model training and application through a series of Python scripts (. py files), covering both class incremental learning networks (Vt class incremental learning networks) and traffic target classification tasks.

2. File Function Description

model_net.py

Function: Implement network model related code, which is the core file for building Vt class incremental learning networks and networks used for traffic target classification (such as ResNet reconstruction networks pre trained on ImageNet). Including network layer definitions, residual blocks, and other infrastructure construction, providing network architecture support for subsequent model training.
Involving key network logic:
Build a network that adapts to the input of 224x224 array pixel grayscale data, including an input layer that receives image information, a 3x3 convolutional layer, a batch normalization (BatchNorm) layer, and a ReLU activation function for feature extraction and processing, and a residual connection (including 1x1 convolutional layer linear projection adaptation dimension) to alleviate gradient vanishing, enhance feature reuse, and help the network converge stably.
Implement EWC related logical foundations for Vt type incremental learning networks, facilitating importance evaluation and regularization constraints on network parameters (such as convolutional layer and batch normalization layer parameters in residual blocks) during the incremental learning phase, balancing old knowledge consolidation and new knowledge acquisition.

Create_dataset.py

Function: Responsible for creating datasets for traffic scenario related tasks (Vt class incremental learning, traffic target classification), constructing datasets that meet model input requirements (such as 224x224 grayscale image data), including data collection, organization, formatting, and other operations, providing data sources for model training.

generate_jiguang_model.py

Function: Contains classes and methods for generating datasets, which can be used to generate and process traffic scene related data, such as simulating and expanding datasets for network training according to specific rules, providing underlying data generation capabilities for data processing processes such as Create_dataset.cy.

train_traffic.py

Function: Implement training and visualization processes for traffic target classification tasks. Based on pre trained convolutional neural networks (such as reconstructed ResNet networks), classify and train 14 types of traffic targets, including traffic signs, pedestrians, and vehicles. Including the construction of a standardized processing pipeline, adaptation to 224x224 grayscale image input, initialization and training of the network using pre trained weights, and visualization functions (such as changes in training process indicators, visualization of classification results, etc.) to assist in analyzing training effectiveness.

train_vds.py

Function: Used for training and visualizing Vt class incremental learning networks. Based on the network architecture constructed by model_net.cy, combined with the dataset prepared by Create_dataset.cy, incremental learning training is carried out. The EWC mechanism is used to balance the learning of new and old task knowledge. During the training process, the convergence of the network and the impact of parameter changes can be visualized to help debug and optimize the network training process.

3. Suggested usage process

pip install -r requirements.txt

python Create_dataset.py

python train_vds.py

python train_traffic.py

