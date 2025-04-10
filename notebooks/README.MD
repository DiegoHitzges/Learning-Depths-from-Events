# Notebook Descriptions

This folder contains four Jupyter Notebooks designed to streamline the training, testing, inference, and visualization of our approach for learning depth from event-based ray densities. The notebooks are configured to be run as they are.  All available options and configurations are explained in detail within the notebooks. The DSIs and LiDAR depthmaps are not included themselves due to storage limitations. To create them, see the associated online directory of the MC-EMVS paper. Please store them in the prepared dataset sub-directories or adjust the associated paths within the notebook accordingly.

### 1. Classes_and_Functions
This notebook defines essential components for training and inference:
- **Dataset Class**: Defines the dataset used for training and testing, with a streamlined version for inference.
- **Neural Network Definition**: Contains the architecture of the neural network.
- **Training and Testing Functions**: Includes functions to handle the network's training and testing.
- **Performance Metrics Function**: Provides a function to compute performance metrics for evaluating model accuracy and efficiency.

### 2. Training_and_Testing
This notebook enables training the network, either from scratch or by continuing from a pretrained model. It also supports testing the network’s performance.

### 3. Inference
This notebook enables inference on trained networks. It provides depth estimates for all frames in the specified dataset and includes options for visualizing these estimates. Inference time is also measured for performance analysis.

### 4. Visualization
In this notebook, a video can be generated for a chosen sequence and filter, displaying:
- **True Depths**
- **Network Depth Estimates**
- **MC-EMVS Depth Estimates**
These visualizations allow for an in-depth comparison of true and estimated depths across multiple configurations.
