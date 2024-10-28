# Learning Depth from Event-based Ray Densities

This is the repository for the conference paper ... The aim is to learn depth from event-camera data. For this, we employ a mix of a geometric and a learning based approach by processing the events into disparity space images (DSIs), filtering these for condfident pixels and then extracting a sub-area of the DSI around it. These Sub-DSIs are then fed into a 3D-Convolutional GRU neural network to estimate depth.

![Alt Text](assets/Framework_cropped.png)

All code of our approach is given in Jupyter Notebooks in the folder ... With these DSIs, models of our architecture can be trained and tested with the notebook ... For inference, see the notebook ... To visualize results, see the notebook ... Each application process is detailed step by step.

### Data-Preprocessing

The events are processed into disparity space images (DSIs), representing the potential depth of each pixel across multiple disparity levels by counting the rays passing through each voxel, projected from the pixel where an event was triggered. In stereo event vision, DSIs from two or more cameras can be fused, eliminating the need for event synchronization between cameras, thus reducing complexity and allowing for more robust depth estimation. To obtain the DSIs, see the repository ...

### Input

From the DSI, we filter for pixels with a sufficiently high ray count. For each of these pixels, a sub-area around them is then extracted from the DSI and normalized. Each of these Sub-DSIs serve as input for the neural nework.

### Neural Network

The neural network is a 3D-Convolutional GRU. Each Sub-DSI undergoes a 3-Convolution to capture geometric patterns and improve generalization. Then, every depth layer is fed sequentially to a GRU. Since the Sub-DSI consist of the counts of rays that have been projected from the representive camera position into space, each depth layer depends on the previous one. Therefore, the last hidden state represents the embedding of all 3D geometric data into a 2D matrix. Two fully connected layers then map to the output of the network.

![Alt Text](assets/neural_net.png)

### Output

We present two versions of the network. They are identical in their architecture, except for their last layer. In the single-pixel version, the network estimates the depth at the selected pixel, central within the Sub-DSI. In the multi-pixel version, the output is a 3x3 grid, also estimating the adjacent pixels depths.

### Results

On the MVSEC indoor flyinig sequence, our approach has outperformed the state-of-the-art (SOTA) DSI-based method by 40% with regards to the MAE. Compared to the SOTA non-DSI approach, we reduced the MAE by over ...%. Moreover, we strongly outperformed all existing learning-based methods.

Hier Video oder Bild einfügen von Vergleich
