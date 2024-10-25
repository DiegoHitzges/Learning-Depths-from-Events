# Learning Depth from Event-based Ray Densities

This is the repository for the conference paper ... The aim is to learn depth from event-camera data. For this, we employ a mix of a geometric and a learning based approach.

The events are processed into disparity space images (DSIs), representing the potential depth of each pixel across multiple disparity levels by counting the rays passing through each voxel, projected from the pixel where an event was triggered. In stereo event vision, DSIs from two or more cameras can be fused, eliminating the need for event synchronization between cameras, thus reducing complexity and allowing for more robust depth estimation.

We then filter each DSI for pixels with a sufficiently high ray count and extract sub-areas around them. These Sub-DSIs are then fed into a 3D-convolutional GRU neural network to obtain pixel-wise depth. Iin a second, multi-pixel network version, depth is estimated for the 3x3 grid surrounding the selected pixel.

Input image here

To obtain the DSIs, see the repository ...

With these DSIs, models of our architecture can be trained and tested with the notebook ... For inference, see the notebook ... To visualize results, see the notebook ... Each application process is detailed step by step.
