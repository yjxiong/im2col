
addpath('./bin');

%% Experiment setup

% Number of channels of input image
channel_in = 32;

% Number of channels of output image
channel_out = 16;

% Size of the sliding windoes/kernel
kernel_size = [5 5];

% Stride in x and y direction
stride = [1,1];

% Input image height-width
img_size = [50 50];

%% Build input image

% Or substitute with your own image matrix
img = single(randn([img_size, channel_in]));


%% Run the CUDA version im2col
cols = cu_im2col(img,kernel_size,stride);


%% err chk
% Run a cpu version of im2col to check the result
cpu_cols = im2colstep(double(gather(img)), [kernel_size,channel_in], [stride,1]);

%% Get the different between 
diff = (cols-cpu_cols);
err = sum(diff(:).^2)

%% Unload the cuda module
clear cu_im2col