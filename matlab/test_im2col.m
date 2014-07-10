
addpath('./bin');

%% Experiment setup

% Number of channels of input image
channel_in = 32;

% Size of the sliding windows/kernels
kernel_size = [7 7];

% Stride in x and y direction
stride = [1,1];

% Input image height-width
img_size = [60 100];

%% Build input image

% Or substitute with your own image matrix
img = single(randn([img_size, channel_in]));


%% Run the CUDA version col2im
cols = cu_im2col(img,kernel_size,stride);


%% err chk
% Run a cpu version of col2im to check the result
% Here we permute the matrix to conform row-major memory order
cpu_cols = im2colstep(double(permute(img, [2 1 3])), [kernel_size,channel_in], [stride,1]);

%% Get the different between 
diff = (cols-cpu_cols);
err = sum(diff(:).^2)

%% Unload the cuda module
clear cu_im2col