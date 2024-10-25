#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# This notebook can be used for training and testing the neural network.
# 
# It relies on the classes and functions defined in *Classes_and_Functions.ipynb*

# In[1]:


# Standard library imports
import random
import os
import gc
import re
import time

# Third-party library imports
import numpy as np
import cv2  # OpenCV for adaptive filtering
import psutil  # For system resource management
from scipy.ndimage import convolve  # To convolve filtering masks

# PyTorch specific imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


# Notebooks
#import import_ipynb
from Classes_and_Functions import *


# # Hyperparameters
# 
# First, define the hyperparameters of which dataset to use, what filter to apply, how Sub-DSIs shall be constructed and the whether to use the single or the multi-pixel version of the network. More options exist for the dataset, see *Classes_and_Functions.ipynb*
# 
# Quick overview:
# * Everything can be left at default except the path for the <b>dsi_directory</b> and the <b>depthmap_directory</b>. 
# * The default is the single-pixel version of the network, to use the multi-pixel version set <b>multi_pixel=True</b>.
# * The process is set to MVSEC stereo on default. If desired, switch to <b>dataset="mvsec_mono"</b> or <b>dataset="dsec"</b>.
# * The filter parameters are set to default, but for MVSEC, we used <b>filter_size=9</b> and an <b>adaptive_threshold_c=-10</b> for training and testing instead. Feel free to replicate.

# In[3]:


"""
Hyperparameters for the dataset:
    # DSI Selection Arguments
    dataset (str): The dataset used.
    test_seq (int): Sequence for testing.
    train_seq_A, train_seq_B (str): Sequences for training.
    dsi_directory (str): Directory of the DSIs. Must be adjusted to user.
    depthmap_directory (str): Directory of the groundtrue depths for each DSI.
    dsi_split (str or int)
    dsi_split (str or int): Which DSIs shall be considered.
                            Can be "all", "even", "odd" or a number between 0 and 9, refering to the last digit of its id.
    dsi_ratio (float): Between 0 and 1. Defines the proportion of (random) DSIs that shall be used.
    start_idx, end_idx (str): Start and stop indices for which DSIs to consider. 
    start_row, end_row, start_col, end_col (str): Define the rows and columns to be considered within each DSI.

    # Pixel selection
    filter_size (int): Determines the size of the neighbourhood area when applying the adaptive threshold filter.
    adaptive_threshold_c (int): Constant that is subtracted from the mean of the neighbourhood pixels when apply the adaptive threshold filter.

    # Sub-DSIs sizes
    sub_frame_radius_h (int): Defines the radius of the frame at the height axis around the central pixel for the Sub-DSI.
    sub_frame_radius_w (int): Defines the radius of the frame at the width axis around the central pixel for the Sub-DSI.

    # Network version
    multi_pixel (bool): Determines whether depth is predicted only for the central selected pixel or for the 8 neighbouring pixels as well.
"""

# Dataset selection
dataset = "mvsec_stereo" #  Options: mvsec_stereo, mvsec_mono, dsec
test_seq = 1  #  Options: 1,2,3 (only for MVSEC sequence)
train_seq_A, train_seq_B = {1,2,3} - {test_seq}

# Directories
dsi_directory_test = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{test_seq}/dsi_stereo/" #  Set your path here
dsi_directory_train_A = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{train_seq_A}/dsi_stereo/"
dsi_directory_train_B = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{train_seq_B}/dsi_stereo/"
depthmap_directory_test = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{test_seq}/depthmaps/" #  Set your path here
depthmap_directory_train_A = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{train_seq_A}/depthmaps/"
depthmap_directory_train_B = f"/mnt/RIPHD4TB/diego/data/mvsec/indoor_flying{train_seq_B}/depthmaps/"

# dsi_split
split = "even"
dsi_split_test = split #  Options: all, even, odd, 0, 1, ..., 9
dsi_split_train_A = split
dsi_split_train_B = split

# dsi_ratio
dsi_ratio_test = 1.0 #  Options: 0 < dsi_ratio <= 1
dsi_ratio_train_A = 1.0
dsi_ratio_train_B = 1.0

# start_idx and end_idx
start_idx_test, end_idx_test = 140-5, 1201-5 #  0, None 
start_idx_train_A, end_idx_train_A = 160-5, 1580-5 #  0, None
start_idx_train_B, end_idx_train_B = 125-5, 1815-5 #  0, None

# start and end idx of rows and columns
start_row_test, end_row_test = 0, None
start_col_test, end_col_test = 0, None
start_row_train_A, end_row_train_A = 0, None
start_col_train_A, end_col_train_A = 0, None
start_row_train_B, end_row_train_B = 0, None
start_col_train_B, end_col_train_B = 0, None

# Filter parameters for pixel selection
filter_size_test = None #  None automatically sets default value. We used 9 for training and testing on MVSEC instead
filter_size_train_A = 9 #  9 for MVSEC
filter_size_train_B = 9 #  9 for MVSEC
adaptive_threshold_c_test = None #  None automatically sets default value. We used -10 for MVSEC instead
adaptive_threshold_c_train_A = -10 #  -10 for MVSEC
adaptive_threshold_c_train_B = -10 #  -10 for MVSEC

# Sub-DSI sizes
sub_frame_radius_h = 3
sub_frame_radius_w = 3

# Network version
multi_pixel = False #  Set to True for multi-pixel network version


# In[4]:


# If DSEC was selected as dataset, decide below whether which half to use for training and testing.
# middle_idx is set to the middle of the index for the zurich_city04a sequence, but can be set to a different custom value as well.
if dataset == "dsec":
    # For DSEC training and test sets can be split by divining one sequence
    middle_idx = 174
    # First half being used for training and second half for testing.
    start_idx_test = middle_idx
    end_idx_train_A = middle_idx
    # Out-comment the 2 lines above and un-comment the 2 lines below to reverse order 
    """
    end_idx_test = middle_idx
    start_idx_train_A = middle_idx
    """
    # A second training set is not needed for DSEC
    end_idx_train_B = start_idx_train_B


# # Training

# ### Datasets

# In[5]:


random.seed(0)
# Decide whether the progress of reading in the DSIs shall be printed for tracking
print_progress = True

# Create training data
training_data_A = DSI_Pixelswise_Dataset(dataset=dataset,
                                         data_seq=train_seq_A,
                                         dsi_directory=dsi_directory_train_A,
                                         depthmap_directory=depthmap_directory_train_A,
                                         dsi_split=dsi_split_train_A,
                                         dsi_ratio=dsi_ratio_train_A,
                                         start_idx=start_idx_train_A, end_idx=end_idx_train_A,
                                         start_row=start_row_train_A, end_row=end_row_train_A,
                                         start_col=start_col_train_A, end_col=end_col_train_A,
                                         filter_size=filter_size_train_A,
                                         adaptive_threshold_c=adaptive_threshold_c_train_A,
                                         sub_frame_radius_h=sub_frame_radius_h,
                                         sub_frame_radius_w=sub_frame_radius_w,
                                         multi_pixel=multi_pixel,
                                         clip_targets=True, #  Clip depths for training
                                         print_progress=print_progress
                                        )

if print_progress: print("")

training_data_B = DSI_Pixelswise_Dataset(dataset=dataset,
                                         data_seq=train_seq_B,
                                         dsi_directory=dsi_directory_train_B,
                                         depthmap_directory=depthmap_directory_train_B,
                                         dsi_split=dsi_split_train_B,
                                         dsi_ratio=dsi_ratio_train_B,
                                         start_idx=start_idx_train_B, end_idx=end_idx_train_B,
                                         start_row=start_row_train_B, end_row=end_row_train_B,
                                         start_col=start_col_train_B, end_col=end_col_train_B,
                                         filter_size=filter_size_train_B,
                                         adaptive_threshold_c=adaptive_threshold_c_train_B,
                                         sub_frame_radius_h=sub_frame_radius_h,
                                         sub_frame_radius_w=sub_frame_radius_w,
                                         multi_pixel=multi_pixel,
                                         clip_targets=True,
                                         print_progress=print_progress
                                        )    


# In[6]:


# Merge training data
training_data = ConcatDataset([training_data_A, training_data_B])
# Inherit some attributes
training_data.dataset = training_data_A.dataset
training_data.pixel_count = training_data_A.pixel_count + training_data_B.pixel_count
training_data.frame_height, training_data.frame_width = training_data_A.frame_height, training_data_A.frame_width
training_data.min_depth, training_data. max_depth = training_data_A.min_depth, training_data_B. max_depth


# In[7]:


# Wrap data into Dataloader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


# In[8]:


# Print data dimensions
training_data_A_size = len(training_data_A)
training_data_B_size = len(training_data_B)
training_data_size = len(training_data)
sub_dsi_size = training_data_A.data_list[0][1].shape

print("training data A size:", training_data_A_size)
print("training data B size:", training_data_B_size)
print("training data size:", training_data_size)
print("pixel number for inference:", training_data.pixel_count)
print("sub dsi size:", sub_dsi_size)


# ### Initialize Model
# 
# More options exist for the network architecture, see *Classes_and_Functions.ipynb*

# In[9]:


# Initialize model
model = PixelwiseConvGRU(sub_frame_radius_h, sub_frame_radius_w, multi_pixel=multi_pixel)
# Send to cuda
if torch.cuda.is_available():
    model.cuda()
# Print architecture
print(model)


# In[10]:


# Define conditions for the training process
epochs = 5 #  3
data_augmentation = False #  data_augmentation randomly inverts DSIs on horizontally and/or vertically
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = CustomMAELoss() if multi_pixel else torch.nn.L1Loss() # CustomMAELoss is L1Loss which ignores  NaN-values


# In[ ]:


# If desired, an previous trained version of the model can be loaded. To do so, give the file name and uncomment this cell.
"""
previous_model_path = "example_path" #  Set your path here
previous_model_file = "example_file" #  Set your file name here
model.load_parameters(previous_model_file, device=device, model_path=previous_model_path, optimizer=optimizer)
""";


# In[11]:


# Set path to store model in directory
model_path = f"/mnt/RIPHD4TB/diego/models/mvsec/indoor_flying{test_seq}/" #  "example_path"
# Define name of model file
model_file = f"{split}_model" #  "example_model"
# In the training process, the current epoch will be added to each files name
# Therefore do NOT set ".pth"
if model_file.endswith(".pth"):
    model_file = model_file[:-4]


# ### Train

# In[12]:


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
    save_points = [int(num_batches * (i / 5)) for i in range(1, 5+1)]
    # Account for single or multi pixel network-version
    num_estims = 9 if model.multi_pixel else 1
    # Track estimates and true depths
    epoch_network_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_argmax_estimates = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    epoch_true_depths = torch.zeros(num_estims * data_size, dtype=torch.float32, device=device)
    # Track current index for these tensors
    current_idx = 0
    
    # Iterate over batches
    save_batch = 0
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
        
        # Save model
        if (batch + 1) in save_points:
            save_batch += 1
            checkpoint_file = f"{epoch_model_file[:-4]}_batch_{save_batch}.pth"
            model.save_model(optimizer, checkpoint_file, model_path=model_path, print_save=True)
        # Clear memory cache
        gc.collect()
            
    # Compute and print performance for epoch
    evaluate_performance(data, epoch_network_estimates, epoch_argmax_estimates, epoch_true_depths)


# In[13]:


# Start the training process
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    # Add epoch to file name
    epoch_model_file = f"{model_file}_epoch_{epoch+1}.pth"
    # Train and track time
    st = time.time()
    train(train_dataloader, training_data, model, loss_fn, optimizer, data_augmentation=data_augmentation)
    ct = time.time()
    print("\n", "Training time [min]: ", (ct-st)//60, sep="")
    # Save
    #model.save_model(optimizer, epoch_model_file, model_path=model_path, print_save=True) #  Set print_save to False to not print message
    print("")
print("Done!")

"""
# # Testing

# ### Dataset

# In[14]:


random.seed(0)
# Decide whether the progress of reading in the DSIs shall be printed for tracking
print_progress = True

# Create testset
test_data = DSI_Pixelswise_Dataset(dataset=dataset,
                                   data_seq=test_seq,
                                   dsi_directory=dsi_directory_test,
                                   depthmap_directory=depthmap_directory_test,
                                   dsi_split=dsi_split_test,
                                   dsi_ratio=dsi_ratio_test,
                                   start_idx=start_idx_test, end_idx=end_idx_test,
                                   start_row=start_row_test, end_row=end_row_test,
                                   start_col=start_col_test, end_col=end_col_test,
                                   filter_size=filter_size_test,
                                   adaptive_threshold_c=adaptive_threshold_c_test,
                                   sub_frame_radius_h=sub_frame_radius_h,
                                   sub_frame_radius_w=sub_frame_radius_w,
                                   multi_pixel=multi_pixel,
                                   clip_targets=False, #  Do not clip depths for testing
                                   print_progress=print_progress
                                  )


# In[15]:


# Wrap data into Dataloader
batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[16]:


# Print data dimensions
test_data_size = len(test_data)
sub_dsi_size = test_data.data_list[0][1].shape

print("test data size:", test_data_size)
print("pixel number for inference:", test_data.pixel_count)
print("sub dsi size:", sub_dsi_size)


# In[ ]:


num_models = 2 #  How many models


# In[ ]:


# Initialize models
models = [PixelwiseConvGRU(sub_frame_radius_h, sub_frame_radius_w, multi_pixel=multi_pixel) for _ in range(num_models)]
# Send to cuda
if torch.cuda.is_available():
    for model in models:
        model.cuda()
# Print architecture
print(models[0])


# In[17]:


# Decide whether test data should be inverted horizontally
flip_horizontal = False
# Decide whether test data should be inverted vertically
flip_vertical = False
# To rotate the data by 0, 90, 180 or 270 degrees, set rotate to 0, 1, 2 or 3.
rotate = 0


# In[ ]:


# Set path to load models from directory
model_paths = [f"/mnt/RIPHD4TB/diego/models/mvsec/indoor_flying{test_seq}/"] * 2#  ["example_path_A, example_path_B"]

for epoch in range(1, epochs+1):
    for batch in range(1,5+1):
        # Give names of model files
        model_files = [f"even_model_epoch_{epoch}_batch_{batch}.pth", f"odd_model_epoch_{epoch}_batch_{batch}.pth"] #  ["example_model_A.pth", "example_model_B.pth"]
        # Do not forget ".pth"
        for idx, model_file in enumerate(model_files):
            if not model_file.endswith(".pth"):
                model_files[idx] += ".pth"
        # Load models parameters
        for idx, model in enumerate(models):
        model.load_parameters(model_files[idx], device=device, model_path=model_paths[idx], optimizer=None)
        # Use ensemble learning to create averaged model
        model = AveragedNetwork(models)
        test(test_dataloader, test_data, model, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, rotate=rotate)        


# In[18]:


test(test_dataloader, test_data, model, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, rotate=rotate)    

"""