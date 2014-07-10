addpath('./bin');

%% Experiment setup

% Number of channels of output image
channel_out = 16;

% Size of the sliding area, default [1,1] to put every element to one pixel
summing_size = [1 1];

% Stride in x and y direction
stride = [1,1];

% Output image height-width
img_size = [60 70];

%% Build input columns

% Or substitute with your own image matrix
cols = single(randn([channel_out, img_size(1)*img_size(2)]));


%% Run the CUDA version col2im
img = cu_col2im(cols, [img_size,channel_out], [summing_size,channel_out]);


%% err chk
% Run a cpu version of im2col to check the result
cpu_img = col2imstep(double(cols), [img_size,channel_out], [summing_size,channel_out]);

%% Get the different between 
diff = (img-single(cpu_img));
err = sum(diff(:).^2)

%% Unload the cuda module
clear cu_col2im