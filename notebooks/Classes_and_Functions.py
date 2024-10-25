#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# In this notebook we will define 
# - the data class to transform DSIs into a format that can be processed by the neural network
# - a streamlined data class used directly for inference
# - the neural network class
# - the metrics and loss function
# - the functions for training process
# - the functions for the testing process
# 
# These methods will then be used an executed in the other notebooks.

# # Dependencies
# 
# We will use PyTorch as frame.

# In[ ]:


# Standard library imports
import random
import os
import gc
import re

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for adaptive filtering
import psutil  # For system resource management
from scipy.ndimage import convolve  # To convolve filtering masks

# PyTorch specific imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


# Get cpu or  gpu device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# # Dataclass 
# 
# For Training, Testing and Visualization.

# In[ ]:


class DSI_Pixelswise_Dataset(Dataset):
    """
    A dataset class to transform DSIs into data for the neural network.
    DSIs are filtered for confident pixels by applying an adaptive threshold filter over the maximum ray counts of each pixel.
    For each of these pixels, a surrounding subregion of the DSI is stored and normalized as a Sub-DSI to be used as data instance.
    The inputs will be these Sub-DSIs, while the targets will be the ground true depth at the according pixels.
    
    Args:
        # DSI Selection Arguments
        dataset (str): The dataset used.
        data_seq (str): The specific sequence of the chosen dataset. Must be adjusted to user.
        dsi_directory (str): Directory of the DSIs. Must be adjusted to user.
        depthmap_directory (str): Directory of the groundtrue depths for each DSI. 
        dsi_split (str or int): Which DSIs shall be considered. Can be "all", "even", "odd" or a number between 0 and 9.
        dsi_ratio (float): Between 0 and 1. Defines the proportion of (random) DSIs that shall be used.
        start_idx, end_idx (str): Start and stop indices for which DSIs to consider. 
        start_row, end_row, start_col, end_col (str): Define the rows and columns to be considered within each DSI.

        # DSI processing
        neg_depth_axis (bool): States whether the depth axis of the DSI has been defined negatively upon creation.
        normalize_dsi (bool): DSIs can be normalized beforehand instead of scaling each Sub-DSIs individually by its own highest ray count.
        
        # Pixel selection
        filter_size (int): Determines the size of the neighbourhood area when applying the adaptive threshold filter.
        adaptive_threshold_c (int): Constant that is subtracted from the mean of the neighbourhood pixels when apply the adaptive threshold filter.
        inference_mode (bool): Flag. If False only pixels with known ground truth depth as targets are stored as data points.

        # Input creation (Sub-DSIs)
        sub_frame_radius_h (int): Defines the radius of the frame at the height axis around the central pixel for the Sub-DSI.
        sub_frame_radius_w (int): Defines the radius of the frame at the width axis around the central pixel for the Sub-DSI.
        center_as_norm_ref (bool): Sub-DSIs can be normalized with respect to highest ray count of central pixel instead of total max value.
        norm_pixel_pos (bool): Defines whether the pixel position should be normalized (to be used as additional input) or not (better for reconstruction).
        
        # Target creation (ground true depth)
        multi_pixel (bool): Determines whether depth is predicted only for the central selected pixel or for the 8 neighbouring pixels as well.
        inverse_space (bool): Defines if target depths and argmax estimates are normalized in linear or inverse space.
        clip_targets (bool): Defines whether targets should be clipped to the believed min and max distance, thus being normalized to inbetween 0 and 1.
    
        # Execution and debugging
        preload_data (bool): Defines whether the data should by loaded directly upon creating a class instance.
        print_progress (bool): Decides whether the progress of creating the data for the class should by displayed.
        debugging (bool): Debugging mode visually prints some data instances as images.
    
    Attributes:
        # Parameters
        frame_height, frame_width (int): The height and width of the frame, equal to the dimensions of the DSIs.
        min_depth, max_depth (int): Set the estimated range of distance for which the DSIs were created.
        max_confidence (int): The maximum relevant ray count within a dataset sequence.
        distCoeffs, K, P (arr): Coefficients to describe the camera lens properties for (un)distortion.
        # Data information
        data_list (list): List of all data instances.
        pixel_count (int): Denotes the total number of pixels for which depth would be predicted (includes pixels without available ground truth).
    
    The attributes are defined by the dataset sequence itself and the way the DSIs were created and should only be changed accordingly.
    """
    def __init__(self,
                 # DSI selection arguments
                 dataset="mvsec_stereo",
                 data_seq=1,
                 dsi_directory=None,
                 depthmap_directory=None,
                 dsi_split="all",
                 dsi_ratio=1.0,
                 start_idx=0, end_idx=None,
                 start_row=0, end_row=None,
                 start_col=0, end_col=None,
                 # DSI processing
                 neg_depth_axis=True,
                 normalize_dsi=False,
                 # Pixel selection
                 filter_size=None,
                 adaptive_threshold_c=None,
                 inference_mode=False,
                 # Input creation (Sub-DSIs)
                 sub_frame_radius_h=3,
                 sub_frame_radius_w=3,
                 center_as_norm_ref=False,
                 norm_pixel_pos=False,
                 # Target creation (ground true depth)
                 multi_pixel=False,
                 inverse_space=False,
                 clip_targets=True,
                 # Execution and debugging
                 preload_data=True,
                 print_progress=False,
                 debugging=False
                ):

        # Args
        self.dataset = dataset
        self.data_seq = data_seq
        self.dsi_split = dsi_split
        self.dsi_ratio = dsi_ratio
        self.neg_depth_axis = neg_depth_axis
        self.normalize_dsi = normalize_dsi
        self.inference_mode = inference_mode
        self.sub_frame_radius_h = sub_frame_radius_h
        self.sub_frame_radius_w = sub_frame_radius_w
        self.center_as_norm_ref = center_as_norm_ref
        self.norm_pixel_pos = norm_pixel_pos
        self.multi_pixel = multi_pixel
        self.inverse_space = inverse_space
        self.clip_targets = clip_targets
        self.preload_data = preload_data
        self.print_progress = print_progress
        self.debugging = debugging
        
        # Default DSI and depthmap directories
        if self.dataset == "mvsec_stereo":
            self.dsi_directory = f"/home/diego/Stereo Depth Estimation/data/mvsec/indoor_flying{self.data_seq}/dsi/"
            self.depthmap_directory = f"/home/diego/Stereo Depth Estimation/data/mvsec/indoor_flying{self.data_seq}/depthmaps/"
        elif self.dataset == "mvsec_mono":
            self.dsi_directory = f"/home/diego/Stereo Depth Estimation/data/mvsec/indoor_flying{self.data_seq}/dsi_mono/"
            self.depthmap_directory = f"/home/diego/Stereo Depth Estimation/data/mvsec/indoor_flying{self.data_seq}/depthmaps/"
        elif self.dataset == "dsec":
            self.dsi_directory = "/home/diego/Stereo Depth Estimation/data/dsec/zurich_city_04_a/dsi/"
            self.depthmap_directory = "/home/diego/Stereo Depth Estimation/data/dsec/zurich_city_04_a/depthmaps/"
        # Use adjusted directory instead if specified
        if dsi_directory is not None:
            self.dsi_directory = dsi_directory
        if depthmap_directory is not None:
            self.depthmap_directory = depthmap_directory
        
        # Assert that file ranges ranges are feasible
        assert self.is_range_feasible(start_idx, end_idx)
        self.start_idx, self.end_idx = start_idx, end_idx
        assert self.is_range_feasible(start_row, end_row)
        self.start_row, self.end_row = start_row, end_row
        assert self.is_range_feasible(start_col, end_col)
        self.start_col, self.end_col = start_col, end_col

        # In debugging mode we only analyze a single DSI
        if self.debugging:
            self.end_idx = min(self.end_idx, self.start_idx + 2)

        # Set filter parameters to default for the data sequence or to the value specified by the argument
        self.filter_size = [5, 5, 5][self.data_seq-1] if self.dataset != "dsec" else 5
        if filter_size is not None:
            self.filter_size = filter_size
        self.adaptive_threshold_c = [-14, -14, -14][self.data_seq-1] if self.dataset != "dsec" else -4
        if adaptive_threshold_c is not None:
            self.adaptive_threshold_c = adaptive_threshold_c
        
        # Attributes:
        self.frame_height, self.frame_width = None, None #  Will be updated with the first DSi
        # estimated depth range
        self.min_depth = [1, 1, 1][self.data_seq-1] if self.dataset != "dsec" else 4
        self.max_depth = [6.5, 6.5, 6.5][self.data_seq-1] if self.dataset != "dsec" else 50
        # maximum relevant ray count
        self.max_confidence = [57.7, 78, 78.8][self.data_seq-1] if self.dataset != "dsec" else 468 
        # camera lens coefficients for undistortion
        self.distCoeffs = np.array([-0.048031442223833355, 0.011330957517194437, -0.055378166304281135, 0.021500973881459395])
        self.K = np.reshape(
            [226.38018519795807, 0.0, 173.6470807871759, 0.0, 226.15002947047415, 133.73271487507847, 0, 0, 1],
            (3, 3)
        )
        self.P = np.reshape(
            [199.6530123165822, 0.0, 177.43276376280926, 0.0, 0.0, 199.6530123165822, 126.81215684365904, 0.0, 0.0, 0.0, 1.0, 0.0],
            (3, 4)
        )

        # Get file names of DSIs and ground true depths
        self.dsi_files = self.get_files()
        self.depthmap_files = self.get_files(depthmaps=True)
        len(self.dsi_files) == len(self.depthmap_files)
        
        # Initialize data list
        self.data_list = []
        self.pixel_count = 0
        # Store ground truth and confidence maps for possible visualization purposes
        if self.inference_mode:
            self.ground_truths = []
            self.confidence_maps = []
        
        # Create Data
        if self.preload_data:
            self.get_data()

    
    """Special Methods:"""
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # pixel_index, sub_dsi, pixel_depth, argmax_depth, frame_idx
        return self.data_list[idx]

    
    """Utility Methods:"""
    def get_files(self, depthmaps=False):
        """Get all file names within selected index-range of DSIs or true depthmaps from the given directory."""
        if not depthmaps:
            # Get DSI files
            directory = self.dsi_directory
            suffix = "_fused.npy" if "mono" not in self.dataset else "_0.npy"
            num_expression = "\d+\.\d+|d+"
        else:
            # Get ground true depthmap files
            directory = self.depthmap_directory
            suffix = ".exr.npy"
            num_expression = "\d+"

        # Load according list of file names from directory
        files = [file for file in os.listdir(directory) if file.endswith(suffix)]
        # Sort files based on number in name
        files.sort(key=lambda file: float(re.findall(num_expression, file)[0]))
        # Select for indices
        files = files[self.start_idx:self.end_idx]
        
        return files

    def get_data(self):
        """Iterate over pixels of DSI and add all selected pixels to the data list."""
        for idx, dsi_file in enumerate(self.dsi_files):
            # Decide whether frame should be processed:
            if self.should_process_dsi(idx):
                # Select pixels within DSI, create the associated data instances and append them to self.data_list
                self.get_pixels(idx, dsi_file)

    def get_pixels(self, idx, dsi_file):
        """
        This is the main method for creating the data.
        It loads the DSI and the associated depthmap.
        Then, a selection filter is applied to the DSI to select pixels that shall become data instances.
        The surrounding Sub-DSIs are created and the true depth values are selected as targets.
        Additionally, the pixel position and the argmax estimation along the depth axis are stored,
        Together, for every selected pixel, all these information are added as one data instance to self.data_list.
        """
        # Track progress in processing the DSIs 
        dsi_idx = self.start_idx + idx
        if self.print_progress:
            print("Load DSI", dsi_idx)
        # Load DSI with threshold_mask
        dsi, threshold_mask = self.get_dsi(dsi_file)
        # Update frame dimensions
        if idx == 0:
            self.frame_height, self.frame_width = dsi.shape[1:]
        
        # Load depthmap with target_mask
        depthmap, target_mask = self.get_depthmap(idx)
        # Create mask to unselect pixels too close to the border to create a Sub-DSI around it
        border_mask = np.zeros_like(threshold_mask)
        border_mask[self.sub_frame_radius_h:-self.sub_frame_radius_h, self.sub_frame_radius_w:-self.sub_frame_radius_w] = True
        # Combine both masks
        selection_mask = border_mask & threshold_mask
        # Count pixels
        if self.multi_pixel:
            # Expand mask to include adjacent pixels
            kernel = np.ones((3,3))
            expanded_mask = convolve(selection_mask, kernel, mode="constant", cval=0)
            # Add pixel count
            self.pixel_count += np.sum(expanded_mask)
        else:
            self.pixel_count += np.sum(selection_mask)
        # Add target available mask for training and numerical evaluation
        if not self.inference_mode:
            selection_mask &= target_mask
        # Deduce indices of selected pixels
        selected_indices = list(zip(*np.where(selection_mask)))
        
        # Get scaled argmax along depth axis as competetive estimate
        argmax_estimates = (dsi.argmax(dim=0) + 1) / dsi.shape[0] 
        # If we use the linear instead of the inverse linear space, the argmax estimates have to be projected accordingly
        if not self.inverse_space:
            # Backproject into original depth space
            argmax_estimates = 1 / (1/self.min_depth - argmax_estimates * (1/self.min_depth - 1/self.max_depth))
            # Project into linear depth space
            argmax_estimates = (argmax_estimates - self.min_depth) / (self.max_depth - self.min_depth)
        
        # Create data for selected pixels and add to self.data_list
        for pixel_index in selected_indices:
            # Get argmax depth estimate for pixel
            argmax_depth = argmax_estimates[pixel_index].clone()
            # Get ground true depth at pixel
            if not self.multi_pixel:
                pixel_depth = depthmap[pixel_index].clone()
            else:
                # For the multi-pixel version we also need the pixel depths at the direct neighbors
                x, y = pixel_index
                pixel_depth = depthmap[x-1 : x+2, y-1 : y+2].flatten()
            # Get sub DSI around selected pixel
            sub_dsi = self.get_sub_dsi(dsi, pixel_index)
            # Normalize pixel position and convert to tensor
            pixel_pos = torch.tensor(pixel_index)
            if self.norm_pixel_pos:
                pixel_pos /= torch.tensor(dsi.shape[1:]) #  Divide through frame_x and frame_y size of DSI
            # Add data to list of data
            pixel_data = (pixel_pos, sub_dsi, pixel_depth, argmax_depth, dsi_idx)
            self.data_list.append(pixel_data)

        # imshow all data steps for debugging
        if self.debugging:
            self.visualize_data_for_debugging(dsi, depthmap, threshold_mask, target_mask, border_mask, selection_mask, sub_dsi)        
        # delete DSI from memory
        del dsi
        gc.collect()

    
    """Helper Methods:"""
    def should_process_dsi(self, dsi_idx):
        """Select whether a DSI should be processed based on the data split and the desired ratio of processed DSIs."""
        # Only select subset of DSIs
        if random.random() > self.dsi_ratio:
            return False
    
        # Select eitehr every DSI, only even/odd ones or only every 10th DSI 
        if self.dsi_split == "all":
            return True
        elif self.dsi_split == "even":
            return dsi_idx % 2 == 0
        elif self.frame_split == "odd":
            return dsi_idx % 2 == 1
        elif self.dsi_split in range(10):
            return dsi_idx % 10 == self.dsi_split
        else:
            raise ValueError("Invalid value for dsi_split. Must be set to 'all', 'even', or 'odd'. Current value: {}".format(self.dsi_split))    

    def get_dsi(self, dsi_file):
        """Load and process DSI. Create adaptive threshold mask based on maximum amount of rays per pixel."""
        # Load (specified area of) DSI as 3d-numpy array
        dsi = np.load(f"{self.dsi_directory}{dsi_file}")[:, self.start_row:self.end_row, self.start_col:self.end_col]
        # Compute threshold mask
        threshold_mask = self.get_threshold_mask(dsi)
        # Normalize DSI (alternatively normalize Sub-DSI)
        if self.normalize_dsi and np.max(dsi) > 0:
            dsi /= np.max(dsi)
        # Transform DSI to pytorch tensor
        dsi = torch.from_numpy(dsi)
        # Flip DSI along depth axis
        if self.neg_depth_axis:
            dsi = dsi.flip(dims=[0])
        return dsi, threshold_mask

    def get_threshold_mask(self, dsi):
        """Create adaptive threshold mask based on maximum amount of counted rays per pixel."""
        # Take the max values of DSI along the depth axis
        confidence_map = np.max(dsi, axis=0)
        if self.inference_mode:
            # Store confidence map
            self.confidence_maps.append(confidence_map)
        # Determine the maximum value for normalization
        dsi_max_confidence = np.max(confidence_map)
        normalization_max_confidence = max(self.max_confidence, dsi_max_confidence)
        # Scale it to inbetween 0 and 255
        confidence_map_normalized = np.around(confidence_map * 255 / normalization_max_confidence).astype('uint8')    
        # Apply adaptive threshold
        threshold_mask = cv2.adaptiveThreshold(confidence_map_normalized, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                self.filter_size, self.adaptive_threshold_c)
        return threshold_mask.astype(bool)

    def get_depthmap(self, idx):
        """Load and preprocess depthmap and create mask for pixels with available ground true depth."""
        # Get depthmap file
        depthmap_file = self.depthmap_files[idx]
        # Load groundtrue depthmap as 2d-numpy array
        depthmap = np.load(f"{self.depthmap_directory}{depthmap_file}")
        # Undistort and eliminate resulting zero values
        if self.dataset != "dsec":
            depthmap = cv2.fisheye.undistortImage(depthmap, self.K, self.distCoeffs, None, self.P)
        # Zero values mean no ground true depth is available
        depthmap[depthmap == 0] = np.nan
        # Select specified area
        depthmap = depthmap[self.start_row:self.end_row, self.start_col:self.end_col]
        # Trunkate values outside of predicted range
        if self.clip_targets:
            depthmap = depthmap.clip(self.min_depth, self.max_depth)
        # Scale to inbetween 0 and 1
        if self.inverse_space:
            # If depth levels shall be projected linearly into inverse space
            depthmap = (1/self.min_depth - 1/depthmap) / (1/self.min_depth - 1/self.max_depth)
        else:
            # If depth levels shall be projected into linear space
            depthmap = (depthmap - self.min_depth) / (self.max_depth - self.min_depth)
    
        # Get target mask (available ground true depth values)
        target_mask = ~np.isnan(depthmap)

        # Store ground truth depth map
        if self.inference_mode:
            self.ground_truths.append(depthmap.copy())
        # Transform to pytorch tensor
        depthmap = torch.from_numpy(depthmap)
    
        return depthmap, target_mask

    def get_sub_dsi(self, dsi, pixel_index):
        """Get sub DSI around selected pixel with frame size of 2*sub_frame_radius + 1."""
        # Get frame borders
        h, w = pixel_index
        sub_frame_h = slice(h - self.sub_frame_radius_h, h + self.sub_frame_radius_h + 1)
        sub_frame_w = slice(w - self.sub_frame_radius_w, w + self.sub_frame_radius_w + 1)
        # Select subregion of DSI
        sub_dsi = dsi[:,sub_frame_h, sub_frame_w].clone()
        # If DSI has not been normalized, normalize on Sub-DSI level
        if not self.normalize_dsi:
            # Normalization can be done either with regards to the highest ray count at the central pixel or the total Sub-DSI
            if self.center_as_norm_ref:
                # Max value at central pixel
                max_val = sub_dsi[:, self.sub_frame_radius_h//2, self.sub_frame_radius_w//2].max()
            else:
                # max value of entire Sub-DSI
                max_val = sub_dsi.max()
            # normalize
            if max_val > 0:
                sub_dsi /= max_val
                
        return sub_dsi

    def is_range_feasible(self, start, end):
        """Assert that start and end indices are feasible."""
        if not isinstance(start, int):
            return False
        if start < 0:
            return False
        if end:
            if not isinstance(end, int):
                return False
            if start > end:
                return False
        return True

    def visualize_data_for_debugging(self, dsi, depthmap, threshold_mask, target_mask, border_mask, selection_mask, sub_dsi):
        """Plot images of data for debugging."""
        import matplotlib.pyplot as plt
        # DSI confidence map
        dsi_max_vals = dsi.numpy().max(axis=0)
        plt.figure(figsize=(16, 12))
        plt.subplot(2,2,1)
        plt.imshow(dsi_max_vals, cmap='Greys')
        plt.colorbar(label='Max Value')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('DSI Confidence Map')
        horizontal = torch.flip(torch.from_numpy(dsi_max_vals), [-1]).numpy()
        plt.subplot(2,2,2)
        plt.imshow(horizontal, cmap='Greys')
        plt.colorbar(label='Max Value')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Horizontally Flipped')
        vertical = torch.flip(torch.from_numpy(dsi_max_vals), [-2]).numpy()
        plt.subplot(2,2,3)
        plt.imshow(vertical, cmap='Greys')
        plt.colorbar(label='Max Value')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Vertically Flipped')
        rotated = torch.flip(torch.from_numpy(dsi_max_vals), [-2, -1]).numpy()
        plt.subplot(2,2,4)
        plt.imshow(rotated, cmap='Greys')
        plt.colorbar(label='Max Value')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Rotated 180 Degrees')
        plt.show()
        # Mask for adaptive threshold
        plt.figure(figsize=(8, 6))
        plt.imshow(threshold_mask, cmap='Greys') #RdYlGn
        plt.colorbar(label='Pixel Selected')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Adaptive Threshold Mask')
        plt.show()
        # Argmax depth estimate map
        argmax_estimates = (dsi.argmax(dim=0) + 1) / dsi.shape[0]
        plt.figure(figsize=(8, 6))
        plt.imshow(argmax_estimates.numpy(), cmap='jet')
        plt.colorbar(label='Depth')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Argmax Depth Estimate')
        plt.show()
        # Ground true depth map
        plt.figure(figsize=(8, 6))
        plt.imshow(depthmap.numpy(), cmap='jet')
        plt.colorbar(label='Depth')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Undistorted Groundtrue Depthmap')
        plt.show()
        # Mask of available groundtrue depths
        plt.figure(figsize=(10, 8))
        plt.imshow(target_mask, cmap='Greys')
        plt.colorbar(label='Pixel Selected')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Available Groundtrue Depths')
        plt.show()
        # Mask eliminating borders
        plt.figure(figsize=(10, 8))
        plt.imshow(border_mask, cmap='Greys')
        plt.colorbar(label='Pixel Selected')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Pixels Within Border')
        plt.show()
        # Combined mask of selected pixels
        plt.figure(figsize=(10, 8))
        plt.imshow(selection_mask, cmap='Greys')
        plt.colorbar(label='Pixel Selected')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Selected Pixels')
        plt.show()
        # Sub-DSI confidence map
        sub_dsi_max_vals = sub_dsi.numpy().max(axis=0)
        plt.figure(figsize=(8, 6))
        plt.imshow(sub_dsi_max_vals, cmap='Greys')
        plt.colorbar(label='Max Value')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Sub DSI Certainty Map')
        plt.show()


# ### Debugging
# 
# Uncomment to use debugging method of data class.

# In[ ]:


# debug_1 = DSI_Pixelswise_Dataset(data_seq=1, start_idx = 500, end_idx = 501, debugging=True)


# In[ ]:


# debug_2 = DSI_Pixelswise_Dataset(data_seq=2, start_idx = 400, end_idx = 401, debugging=True)


# In[ ]:


# debug_3 = DSI_Pixelswise_Dataset(data_seq=3, start_idx = 600, end_idx = 601, debugging=True)


# # Inference-Dataclass
# 
# Streamlined, soly for inference for a given list of DSIs and possibly given adpative gaussian threshold filters.

# In[ ]:


def load_dsi_list(dsi_directory, start_idx=0, end_idx=None):
    """Function to load all DSIs from a directory, starting and ending at a given index."""
    # Name expression of files
    suffix = "_fused.npy"
    num_expression = "\d+\.\d+|d+"
    # Load according list of file names from directory
    files = [file for file in os.listdir(dsi_directory) if file.endswith(suffix)]
    # Sort files based on number in name
    files.sort(key=lambda file: float(re.findall(num_expression, file)[0]))
    # Select for indices
    files = files[start_idx:end_idx]
    # Load DSIs from their files
    dsi_list = [np.load(f"{dsi_directory}{dsi_file}") for dsi_file in files]

    return dsi_list


# In[ ]:


def get_threshold_mask(dsi, filter_size, adaptive_threshold_c, max_confidence):
    """
    Function to create an adaptive threshold mask based on maximum amount of counted rays per pixel.
    Args:
        dsi (numpy arr): A DSI of dimensions depth, height, width
        filter_size (int): Determines the size of the neighbourhood area when applying the adaptive threshold filter.
        adaptive_threshold_c (int): Constant that is subtracted from the mean of the neighbourhood pixels when apply the adaptive threshold filter.
        max_confidence (int): The maximum relevant ray count in the DSI sequence.
    """
    # Take the max values of DSI along the depth axis
    confidence_map = np.max(dsi, axis=0)
    # Determine the maximum value for normalization
    dsi_max_confidence = np.max(confidence_map)
    normalization_max_confidence = max(max_confidence, dsi_max_confidence)
    # Scale it to inbetween 0 and 255
    confidence_map_normalized = np.around(confidence_map * 255 / normalization_max_confidence).astype('uint8')    
    # Apply adaptive threshold
    threshold_mask = cv2.adaptiveThreshold(confidence_map_normalized, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                            filter_size, adaptive_threshold_c)
    return threshold_mask.astype(bool)


# In[ ]:


class Estimated_Depthmaps():
    """
    A dataset class to create the estimated depthmaps as inference from a given list of DSIs.
    The adaptive gaussian threshold filter is either given or automatically computed along the way.
    The depthmaps are then stored as a list of normalized numpy arrays under self.estimated_depths.
    These depthmaps can be colored. For this, call the method self.create_colored_depth_estimations().
    The colored depthmaps are then stored under self.colored_depth_estimations.

    Args:
        model (torch.nn.Module): A trained network model.
        dsi_list (list of numpy arrs): A list of DSIs with dimensions (depth, height, width).
        # Pixel selection
        threshold_mask_list (list of numpy arrs): A list of adaptive gaussian threshold filters for each DSI.
        filter_size (int): Determines the size of the neighbourhood area when applying the adaptive threshold filter.
        adaptive_threshold_c (int): Constant that is subtracted from the mean of the neighbourhood pixels when apply the adaptive threshold filter.
        max_confidence (int): The maximum relevant ray count in the DSI sequence.
        # Input creation (Sub-DSIs)
        sub_frame_radius_h (int): Defines the radius of the frame at the height axis around the central pixel for the Sub-DSI.
        sub_frame_radius_w (int): Defines the radius of the frame at the width axis around the central pixel for the Sub-DSI.
        batch_size (int): The batch size for the Dataloader when applying the network.
        
    Attributes:
        # Parameters
        frame_height, frame_width (int): The height and width of the frame, equal to the dimensions of the DSIs.
        estimated_depths (list of numpy arrs): A list of the estimated depthmaps by the model for each DSI.
        colored_depth_estimations (list of numpy arrs): A colored version of the estimated_depths list. Cmap is 'jet'.
    """
    def __init__(self,
                 model,
                 dsi_list,
                 threshold_mask_list,
                 # Input creation (Sub-DSIs)
                 sub_frame_radius_h=3,
                 sub_frame_radius_w=3,
                 batch_size=1024,
                 ):
        # Args and Attrbts
        self.model = model
        self.dsi_list = dsi_list
        self.threshold_mask_list = threshold_mask_list
        self.sub_frame_radius_h = sub_frame_radius_h
        self.sub_frame_radius_w = sub_frame_radius_w
        self.batch_size = batch_size
        self.frame_height, self.frame_width = self.dsi_list[0].shape[1:]
        # Estimated Depthmaps
        self.estimated_depths = [np.full((self.frame_height, self.frame_width), np.nan) for _ in self.dsi_list]
        # Colored images
        self.colored_depth_estimations = []        
        # Automatically send model to the available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.device = self.device
        model.to(model.device)  # Send the model to the device

        # Iterate through list of DSIs, get threshold mask, transform to data, apply network, assign estimated depths to pixels
        for dsi_idx, dsi in enumerate(self.dsi_list):
            threshold_mask = threshold_mask_list[dsi_idx]
            # Select pixels from DSI and create Sub-DSIs around them as data points
            data_for_inference = DSI_Pixels_for_Inference(dsi, threshold_mask, self.sub_frame_radius_h, self.sub_frame_radius_w)
            # Load data into DataLoader to better parallelization
            dataloader = DataLoader(data_for_inference, batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                # Iterate through batches to apply network and assign estimated depths
                for batch, batch_data in enumerate(dataloader):
                    pixel_position, network_depth = self.apply_network(batch_data)
                    self.assign_pixel_depth(dsi_idx, pixel_position, network_depth)
            gc.collect()

    
    """Methods"""
    def get_threshold_mask(self, dsi):
        """Create adaptive threshold mask based on maximum amount of counted rays per pixel."""
        # Take the max values of DSI along the depth axis
        confidence_map = np.max(dsi, axis=0)
        # Determine the maximum value for normalization
        dsi_max_confidence = np.max(confidence_map)
        normalization_max_confidence = max(self.max_confidence, dsi_max_confidence)
        # Scale it to inbetween 0 and 255
        confidence_map_normalized = np.around(confidence_map * 255 / normalization_max_confidence).astype('uint8')    
        # Apply adaptive threshold
        threshold_mask = cv2.adaptiveThreshold(confidence_map_normalized, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                self.filter_size, self.adaptive_threshold_c)
        return threshold_mask.astype(bool)

    def apply_network(self, batch_data):
        """Apply network to batch data."""
        # Get batch data
        batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
        pixel_position, sub_dsi = batch_data
        # Normalize pixel position
        norm_pixel_position = pixel_position.clone() / torch.tensor((self.frame_height, self.frame_width))
        # Create input for model
        input = (norm_pixel_position, sub_dsi)
        # Compute prediction
        network_depth = self.model(input)
        # Clip network estimations to inbetween 0 and 1
        network_depth = network_depth.clip(0,1)

        return pixel_position, network_depth

    def assign_pixel_depth(self, dsi_idx, pixel_position, network_depth):
        """Assign estimated depths to the given pixels of the current DSI."""
        # Assign estimated depths to each pixel position
        for pixel_idx, pixel_depth in enumerate(network_depth):
            # Get height and width position of individual pixel
            h, w = pixel_position[pixel_idx]
            # Single value
            if not self.model.multi_pixel:
                self.estimated_depths[dsi_idx][h,w] = pixel_depth.item()
            # Multiple values in 3x3 grid
            else:
                i = 0
                # Iterate over left, right, top and down neighbours
                for row in range(h - 1, h + 2):
                    for col in range(w - 1, w + 2):
                        self.estimated_depths[dsi_idx][row, col] = pixel_depth[i].item()
                        i += 1
    
    def create_colored_depth_estimations(self):
        """Color the estimated depthmaps with the cmap 'jet'."""
        # Copy the jet colormap and set color for NaN values
        cmap = plt.colormaps["jet"]
        cmap.set_bad(color='white')
        # Apply colormap
        self.colored_depth_estimations = [cmap(np.ma.masked_invalid(depthmap)) for depthmap in self.estimated_depths]


# In[ ]:


class DSI_Pixels_for_Inference(Dataset):
    """
    A dataset class to transform a single DSI into data for the neural network.
    This class is for inference of a single DSI only. It is a streamlined version of the DSI_Pixelswise_Dataset class.
    The DSI is filtered for confident pixels by applying an adaptive threshold filter, created over the maximum ray counts of each pixel.
    For each of these pixels, a surrounding subregion of the DSI is stored and normalized as a Sub-DSI to be used as data instance.
    These instances are the inputs to the network model.

    Args:
        dsi (numpy arr): The DSI with dimensions (depth, height, width).
        threshold_mask (numpy arr): An adaptive gaussian threshold filter to select pixels with high confidence due to their maximum ray count.
        sub_frame_radius_h (int): Defines the radius of the frame at the height axis around the central pixel for the Sub-DSI.
        sub_frame_radius_w (int): Defines the radius of the frame at the width axis around the central pixel for the Sub-DSI.
    """
    def __init__(self, dsi, threshold_mask, sub_frame_radius_h, sub_frame_radius_w):
        # Args
        self.threshold_mask = threshold_mask
        self.sub_frame_radius_h = sub_frame_radius_h
        self.sub_frame_radius_w = sub_frame_radius_w
        # Initialize data list
        self.data_list = []
        
        # Filter for confident pixels
        selected_indices = self.get_indices()
        # Transform DSI
        dsi = self.transform_dsi(dsi)
        # Create data for selected pixels and add to self.data_list
        for pixel_index in selected_indices:
            # Get sub DSI around selected pixel
            sub_dsi = self.get_sub_dsi(dsi, pixel_index)
            # Convert pixel position to tensor
            pixel_pos = torch.tensor(pixel_index)
            # Add data to list of data
            pixel_data = (pixel_pos, sub_dsi)
            self.data_list.append(pixel_data)

    
    """Special Methods"""
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # pixel_index, sub_dsi
        return self.data_list[idx]
        
    
    """Utility Methods"""
    def get_indices(self):
        """Select confident pixels."""
        # Create mask to unselect pixels too close to the border to create a Sub-DSI around it
        border_mask = np.zeros_like(self.threshold_mask)
        border_mask[self.sub_frame_radius_h:-self.sub_frame_radius_h, self.sub_frame_radius_w:-self.sub_frame_radius_w] = True
        # Combine with threshold masks
        selection_mask = border_mask & self.threshold_mask
        # Deduce indices of selected pixels
        selected_indices = list(zip(*np.where(selection_mask)))
        return selected_indices
    
    def transform_dsi(self, dsi):
        """Transform DSI."""
        # Transform DSI to pytorch tensor
        dsi = torch.from_numpy(dsi)
        # Flip DSI along depth axis
        dsi = dsi.flip(dims=[0])
        return dsi
    
    def get_sub_dsi(self, dsi, pixel_index):
        """Get sub DSI around selected pixel with frame size of 2*sub_frame_radius + 1."""
        # Get frame borders
        h, w = pixel_index
        sub_frame_h = slice(h - self.sub_frame_radius_h, h + self.sub_frame_radius_h + 1)
        sub_frame_w = slice(w - self.sub_frame_radius_w, w + self.sub_frame_radius_w + 1)
        # Select subregion of DSI
        sub_dsi = dsi[:,sub_frame_h, sub_frame_w].clone()        
        # Normalize Sub-DSI
        if sub_dsi.max() > 0:
            sub_dsi /= sub_dsi.max()
        return sub_dsi


# # Neural Network
# 
# We define the neural network architecture as a class with two methods to save and load parameters.
# We also define an additional class that averages the estimates of several of our network to leverage ensemble learning.

# In[ ]:


class PixelwiseConvGRU(nn.Module):
    """
    A neural network class to predict pixel-wise depth.
    Input: Sub-DSI
    Output: Depth estimate for central pixel and, if multi-pixel is set to True, also of the 8 dircetly neighboring pixels
    Architecture: 3D-Convolution -> Flatten -> GRU -> Final hidden state -> Dense layer -> Output.
    
    Args:
        sub_frame_radius_h (int): Radius at the length axis of the frame of the Sub-DSI.
        sub_frame_radius_w (int): Radius at the length axis of the frame of the Sub-DSI.
        out_channels (int): Number of output channels for the 3D-convolution.
        multi_pixel (bool): Decides whether depth shall be estimated only for the central pixel or also at the 8 neighboring pixels.
        use_pixel_pos (bool): An option to append the pixel coordinates to the data vector after the GRU for additional information.
                                Pixel positions must be normalized herefor by the DSI_Pixelswise_Dataset.
        hidden_size_scale (int): A scaling factor to scale the size of the inputs for the GRU to the size of the hidden states.
        num_gru_layers (int): Defines how many GRU layers should be stacked sequentially.
        bidirectional (bool): Defines whether the GRU layer(s) should work bidirectionally.
        dropout_rate (float): Rate for dropout.
    """
    
    def __init__(self,
                 sub_frame_radius_h,
                 sub_frame_radius_w,
                 out_channels=4,
                 multi_pixel=False,
                 use_pixel_pos=False,
                 hidden_size_scale=1,
                 num_gru_layers=1,
                 bidirectional=False,
                 dropout_rate=0
                ):
        # Inherit
        super(PixelwiseConvGRU, self).__init__()
    
        # Args
        self.sub_frame_radius_h = sub_frame_radius_h
        self.sub_frame_radius_w = sub_frame_radius_w        
        # The size of the Sub-DSI frame is 2 times its radius plus the central pixel
        self.sub_frame_size_h = 2 * sub_frame_radius_h + 1
        self.sub_frame_size_w = 2 * sub_frame_radius_w + 1
        self.out_channels = out_channels
        self.multi_pixel = multi_pixel
        self.use_pixel_pos = use_pixel_pos
        self.hidden_size_scale = hidden_size_scale
        self.num_gru_layers = num_gru_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        # Deduct sizes
        self.gru_input_size = self.out_channels * (self.sub_frame_size_h-2) * (self.sub_frame_size_w-2)  # Frame size is reduced since we do not apply padding
        self.gru_hidden_size = self.gru_input_size * self.hidden_size_scale
        self.output_dim = 1 if not self.multi_pixel else 9
        
        # 3D-convolution layer
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                # Pad only along the depth dimension
                # since ray counts are effectively zero for the padded depth levels
                padding=(1, 0, 0), 
                stride=(2,1,1)
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
            )
        
        # GRU layer
        self.gru = nn.GRU(
            input_size = self.gru_input_size,
            hidden_size = self.gru_hidden_size,
            num_layers = self.num_gru_layers,
            dropout = self.dropout_rate,
            bidirectional=self.bidirectional,
            batch_first = True
            )

        # Output layer
        self.dense_output = nn.Sequential(
            nn.Linear(
                # A bidircetional GRU would have double the output size
                # and if the pixel position shall be considered, two entries will be appended
                (1+self.bidirectional)*self.gru_hidden_size + 2*self.use_pixel_pos, self.gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.gru_hidden_size, self.output_dim)
            )

        # Automatically send model to the available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Send the model to the device
        
    def forward(self, input):
        # Preprocess input
        pixel_position, sub_dsi = input
        batch_size, depth_levels = sub_dsi.shape[:2]
        
        # Apply 3D-convolution
        sub_dsi_conv = self.conv3d(sub_dsi.unsqueeze(dim=1))
        # Flatten
        sub_dsi_conv_flat = sub_dsi_conv.transpose(1,2).flatten(start_dim=2)
        # Check whether dimensions match from 3D-convolution to GRU
        batch_size, depth_levels, tensor_size = sub_dsi_conv_flat.size()
        assert tensor_size == self.gru_input_size
        
        # Apply GRU
        h_seq, _ = self.gru(sub_dsi_conv_flat)
        # Take final hidden state
        h_n = h_seq[:,-1,:]
        # If selected, appenid pixel position
        if self.use_pixel_pos:
            h_n = torch.cat([pixel_position, h_n], dim=-1)
        # Check whether dimensions match from GRU output to the final dense output-layer
        assert h_n.size() == (batch_size, (1+self.bidirectional)*self.gru_hidden_size + 2*self.use_pixel_pos)

        # Apply final dense layer to obtain final estimate
        output = self.dense_output(h_n)
        # Assert correct output dimension
        assert output.size() == (batch_size, self.output_dim)
        # Squeeze if single pixel
        if not self.multi_pixel:
            output = output.squeeze(dim=-1)
        
        return output

    def save_model(self, optimizer, model_file, model_path=None, print_save=True):
        """Method to save model and optimizer parameters to model_path and model_file."""
        if model_path is None:
            # Set default model path
            model_path = "/home/diego/Stereo Depth Estimation/models/"
            
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()},
            os.path.join(model_path, model_file)
                  )
        # Print success message
        if print_save:
            print(f"Saved PyTorch Model and Optimizer State to {model_path}{model_file}")

    def load_parameters(self, model_file, device, model_path=None, optimizer=None):
        """Method to load model parameters from model_path and model_file.
        If an optimizer is selected, its parameters are loaded, too.
        """
        if model_path is None:
            # Set default model path
            model_path = "/home/diego/Stereo Depth Estimation/models/"
        checkpoint = torch.load(os.path.join(model_path, model_file), map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# In[ ]:


class AveragedNetwork(nn.Module):
    """Neural network class to average predictions of several networks."""
    def __init__(self, neural_nets):
        super(AveragedNetwork, self).__init__()
        # Inherit
        self.multi_pixel = neural_nets[0].multi_pixel
        self.sub_frame_radius_h = neural_nets[0].sub_frame_radius_h
        self.sub_frame_radius_w = neural_nets[0].sub_frame_radius_w
        
        # Automatically send averaged model to the available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Send the model to the device
        
        # List of neural networks as argument
        if torch.cuda.device_count() > 1:
            # Use DataParallel if multiple GPUs are available
            self.neural_nets = nn.ModuleList([nn.DataParallel(net) for net in neural_nets])
        else:
            # List of neural networks as argument
            self.neural_nets = neural_nets

    def forward(self, input):
        # Forward pass through all networks
        outputs = [neural_net(input) for neural_net in self.neural_nets]
        
        # Calculate the average prediction
        average_output = sum(outputs) / len(outputs)

        return average_output


# # Loss Function
# 
# We write a loss function that ignores NaN-values.

# In[ ]:


class CustomMAELoss(nn.Module):
    """Custom loss function to apply L1-loss but ignore Nan-values."""
    def __init__(self):
        super(CustomMAELoss, self).__init__()

    def forward(self, prediction, target):
        # Check and flatten inputs if necessary
        if prediction.dim() > 2:
            prediction = prediction.flatten(start_dim=1)
        if target.dim() > 2:
            target = target.flatten(start_dim=1)

        # Ensure the prediction and target tensors have the same shape
        assert prediction.shape == target.shape, "Prediction and target must have the same shape"

        # Compute mask to ignore NaNs
        valid_mask = ~torch.isnan(target)
        valid_predictions = prediction[valid_mask]
        valid_targets = target[valid_mask]

        # Calculate the absolute errors only on valid (non-NaN) entries
        abs_errors = torch.abs(valid_predictions - valid_targets)

        # Return the mean of these errors
        return torch.mean(abs_errors)


# # Metrics
# 
# We define an evaluation method to measure performance by computing metrics.

# In[ ]:


def evaluate_performance(data, network_estimates, argmax_estimates, true_depths):
    """
    Evaluates performance of network and argmax approach against true depths.
    Estimates are brought into the right form and then used to compute metrics.
    The data instance itself needs to be passed to derive hyperparameters.
    """
    # Eliminate nan values
    valid_mask = ~torch.isnan(true_depths)
    network_estimates = network_estimates[valid_mask]
    argmax_estimates = argmax_estimates[valid_mask]
    true_depths = true_depths[valid_mask]
    
    # Project values to original space in meters
    network_estimates = network_estimates * (data.max_depth - data.min_depth) + data.min_depth
    argmax_estimates = argmax_estimates * (data.max_depth - data.min_depth) + data.min_depth
    true_depths = true_depths * (data.max_depth - data.min_depth) + data.min_depth

    # Get hyperparameters for camera and scene
    if data.dataset in ["mvsec", "mvsec_stereo", "mvsec_mono"]:
        b, f = 0.09988137641750752, 226.38018519795807
    elif data.dataset=="dsec":
        b, f = 0.6, 557.2412109375
    
    # Compute absolute network performance for epoch
    errors_names = ["MAE", "MedAE", "Bad Pix", "SILog", "ARE", "log RMSE", "delta1", "delta2", "delta3"]
    network_errors = [error_value for error_value in compute_metrics(network_estimates, true_depths, b, f)]
    argmax_errors = [error_value for error_value in compute_metrics(argmax_estimates, true_depths, b, f)]

    # Scale distance errors to centimeters and quotients to percentages
    for i, error_name in enumerate(errors_names):
        network_errors[i] *= 100
        argmax_errors[i] *= 100
    
    # Create output string
    network_string = "Network Test Error Performance:\n"
    argmax_string = "Argmax Test Error Performance:\n"
    for i, error_name in enumerate(errors_names):
        network_string += f" {error_name}: {network_errors[i].item():>0.2f} |"
        argmax_string += f" {error_name}: {argmax_errors[i].item():>0.2f} |"
    # Add number of points for inference
    network_string += f" #Pix: {data.pixel_count}"
    argmax_string += f" #: {data.pixel_count}"
    
    # Print performance
    print(network_string)
    print(argmax_string)


# In[ ]:


def compute_metrics(estimate, target, b, f):
    """Compute metrics given estimates and true depth targets."""
    # Data size
    n = len(estimate)
    # Epsilon to avoid division by zero
    epsilon = 0.00000000001
    estimate += epsilon
    target += epsilon
    
    # MAE
    MAE = torch.mean(torch.abs(estimate - target))
    # MedAE
    MedAE = torch.median(torch.abs(estimate - target))
    # Bad pix
    err = torch.abs(1 / estimate - 1 / target) * b * f
    rel_err = err * target / b / f
    badp = torch.sum((err > 5) & (rel_err > 0.05)) / n
    # SILog
    di = torch.log(target) - torch.log(estimate)
    SILog = 1 / n * torch.sum(di ** 2) - 1 / (n * n) * torch.sum(di) ** 2
    # Abs rel diff error
    ARE = 1 / n * torch.sum(torch.abs(estimate - target) / estimate)
    # log RMSE
    lRMSE = (1 / n * torch.sum((torch.log(target) - torch.log(estimate)) ** 2)) ** 0.5
    # Inlier ratios
    delta = torch.max(estimate / target, target / estimate)
    delta1 = torch.sum(delta < 1.25) / n
    delta2 = torch.sum(delta < 1.25 ** 2) / n
    delta3 = torch.sum(delta < 1.25 ** 3) / n

    return MAE, MedAE, badp, SILog, ARE, lRMSE, delta1, delta2, delta3


# # Training
# 
# We define the training process. Within it, there is a training loop for every batch. Both are defined by the two subsequent functions.

# In[ ]:


def train(dataloader, data, model, loss_fn, optimizer, data_augmentation=False):
    """
    Function to define the training process.
    The data instance itself is needed to derive hyperparameters of the dataset.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set model to training mode
    model.train()
    # Get size of entire dataset
    data_size = len(dataloader.dataset)
    # Get number of batches
    num_batches = len(dataloader)
    # Account for single or multi pixel network-version
    num_estims = 9 if model.multi_pixel else 1
    # Track estimates and true depths
    epoch_network_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_argmax_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_true_depths = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    # Track current index for these tensors
    current_idx = 0
    
    # Iterate over batches
    for batch, batch_data in enumerate(dataloader):
        # If available, use GPU (device has to be set earlier)
        batch_data = tuple(tensor.to(device) for tensor in batch_data)
        # Get batch data
        pixel_position, sub_dsi, true_depth, argmax_depth, frame_idx = batch_data
        batch_size = true_depth.size(0)
        # Train on batch and return network prediction (without augmented predictions)
        pred = train_batch(batch_data, model, loss_fn, optimizer, data_augmentation=data_augmentation)
        # Clip network estimations to inbetween 0 and 1
        network_depth = pred.clip(0,1)
        # Update epoch estimates and target values
        epoch_network_estimates[current_idx:current_idx + num_estims * batch_size] = network_depth.flatten()
        epoch_argmax_estimates[current_idx:current_idx + num_estims * batch_size] = argmax_depth.repeat_interleave(num_estims)
        epoch_true_depths[current_idx:current_idx + num_estims * batch_size] = true_depth.flatten()
        # Update index
        current_idx += num_estims * batch_size
        # Clear memory cache
        gc.collect()
            
    # Compute and print performance for epoch
    evaluate_performance(data, epoch_network_estimates, epoch_argmax_estimates, epoch_true_depths)


# In[ ]:


def train_batch(batch_data, model, loss_fn, optimizer, data_augmentation=False):
    """The iteration step for training on each batch."""
    # Get data
    pixel_position, sub_dsi, true_depth, argmax_depth, frame_idx = batch_data
    # Augment data
    if data_augmentation:
        pixel_position = pixel_position.clone()
        # Invert x-axis
        if random.random() > 0.5:
            pixel_position[:,0] = 1 - pixel_position[:,0]
            sub_dsi = sub_dsi.flip([-2])
        # Invert y-axis
        if random.random() > 0.5:
            pixel_position[:,1] = 1 - pixel_position[:,1]
            sub_dsi = sub_dsi.flip([-1])
    # Define input
    input = (pixel_position, sub_dsi)
    
    # Compute prediction loss
    pred = model(input)
    loss = loss_fn(pred, true_depth)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return pred


# # Testing
# 
# We now define the testing process.

# In[ ]:


def test(dataloader, data, model, flip_horizontal=False, flip_vertical=False, rotate=0):
    """
    Function to test the performance of the model.
    The data instance itself has to be given to derive hyperparameters.
    Data augmentation can be applied by flipping the data horizontally or vertically.
    To rotate the data by 0, 90, 180 or 270 degrees, set rotate to 0, 1, 2 or 3.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.to(device)
    model.eval()
    # Get size of entire dataset
    data_size = len(dataloader.dataset)
    # Get number of batches
    num_batches = len(dataloader)
    # Account for single or multi pixel network-version
    num_estims = 9 if model.multi_pixel else 1
    # Track estimates and true depths
    epoch_network_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_argmax_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_true_depths = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    # Track current index for these tensors
    current_idx = 0
    
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):    
            # If available, use GPU (device has to be set earlier)
            batch_data = tuple(tensor.to(device) for tensor in batch_data)
            # Get batch data
            pixel_position, sub_dsi, true_depth, argmax_depth, frame_idx = batch_data
            batch_size = true_depth.size(0)
            # Rotate and/or mirror data
            if flip_horizontal:
                sub_dsi = sub_dsi.flip([-1])
            if flip_vertical:
                sub_dsi = sub_dsi.flip([-2])
            if rotate > 0:
                sub_dsi = torch.rot90(sub_dsi, k=rotate, dims=[-1, -2])             
            # Get input
            input = (pixel_position, sub_dsi)
            # Compute prediction
            network_depth = model(input)
            # Clip network estimations to inbetween 0 and 1
            network_depth = network_depth.clip(0,1)
            # Update epoch estimates and target values
            epoch_network_estimates[current_idx:current_idx + num_estims * batch_size] = network_depth.flatten()
            epoch_argmax_estimates[current_idx:current_idx + num_estims * batch_size] = argmax_depth.repeat_interleave(num_estims)
            epoch_true_depths[current_idx:current_idx + num_estims * batch_size] = true_depth.flatten()
            # Update index
            current_idx += num_estims * batch_size
    
    # Compute and print performance for epoch
    evaluate_performance(data, epoch_network_estimates, epoch_argmax_estimates, epoch_true_depths)

