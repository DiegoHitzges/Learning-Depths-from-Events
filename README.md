# Learning Depth from Event-based Ray Densities

This repository accompanies our conference paper, "Learning Depth from Event-based Ray Densities." The objective is to estimate depth from event-camera data using a blend of geometric and learning-based approaches. Events are first processed into disparity space images (DSIs), from which pixels with high confidence are selected. Around each of these confident pixels, we extract a local subregion of the DSI (Sub-DSI). These Sub-DSIs are then passed to a 3D-Convolutional GRU neural network, which estimates pixel-wise depth, achieving significant improvements over prior state-of-the-art methods.

![Alt Text](assets/Framework_cropped.png)

All code of our approach is given in Jupyter Notebooks in the folder ... With these DSIs, models of our architecture can be trained and tested with the notebook ... For inference, see the notebook ... To visualize results, see the notebook ... Each application process is detailed step by step.

### Data-Preprocessing

The events are processed into disparity space images (DSIs), which represent the potential depth of each pixel across multiple disparity levels by counting the rays passing through each voxel, projected from the pixel where an event was triggered. In stereo event vision, DSIs from two or more cameras can be fused, eliminating the need for event synchronization between cameras. This reduces complexity and enables more robust depth estimation. This approach was originally proposed in the [MC-EMVS](https://onlinelibrary.wiley.com/doi/10.1002/aisy.202200221) paper. To construct DSIs, refer to the associated repository, [dvs_mcemvs](https://github.com/tub-rip/dvs_mcemvs).

### Input

To ensure reliable depth estimation, we apply an adaptive Gaussian threshold filter to each DSI, filtering for pixels with a sufficiently high ray count. For each selected pixel, a local sub-area around it is extracted from the DSI and normalized. These extracted regions, or Sub-DSIs, serve as the input for the neural network.

### Neural Network

The neural network architecture is a 3D-Convolutional GRU. Each Sub-DSI first undergoes a 3D convolution to capture geometric patterns and enhance generalization. Following this, each depth layer is fed sequentially to a GRU. Because the Sub-DSI consists of ray counts projected from the representative camera position into space, each depth layer depends on the previous one. Consequently, the last hidden state captures the embedding of all 3D geometric data in a 2D matrix. Finally, two fully connected layers map this embedding to the network output.

![Alt Text](assets/neural_net.png)

### Output

We present two versions of the network, identical in architecture except for their last layer. In the single-pixel version, the network estimates the depth at the selected pixel, positioned centrally within the Sub-DSI. In the multi-pixel version, the output is a 3x3 grid, providing depth estimates for the selected pixel as well as its adjacent pixels.

### Results

On the MVSEC indoor flyinig sequence, our approach has outperformed the state-of-the-art (SOTA) DSI-based method by 40% with regards to the MAE. Compared to the SOTA non-DSI approach, we reduced the MAE by over ...%. Moreover, we strongly outperformed all existing learning-based methods.

Hier Video oder Bild einfügen von Vergleich
