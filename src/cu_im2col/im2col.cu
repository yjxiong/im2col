#include "mex.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include "hemi\hemi.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		char err_str[1000];
		sprintf(err_str,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		mexErrMsgTxt(err_str);
		
	}
}

#define HEMI_GRID_STRIDE_LOOP(iter, num) 	for (int iter = hemiGetElementOffset(); \
	iter<num;\
	iter+=hemiGetElementStride())


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(op_idx, n) {
	int index = op_idx;
    int w_out = index % width_col;

    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
	
    float* temp_col = data_col+ (channel_out * height_col + h_out) * width_col + w_out;
    const float* temp_img = data_im + (channel_in * height + h_in) * width + w_in;
	
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *temp_col = (h >= 0 && w >= 0 && h < height && w < width) ?
            temp_img[i * width + j] : 0;
        temp_col += height_col * width_col;
      }
    }
  }
}

#define THREAD_PER_BLOCK 256

#define IMG_OUT plhs[0]

#define IMG_IN prhs[0]
#define KSIZE_IN prhs[1]
#define STRIDE_IN prhs[2]
#define PADDING prhs[3]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//All code and internal function calls go in here!
	if(nrhs<2)
		mexErrMsgTxt("Not enough inputs");

	double* stride_in;
	size_t stride_c, stride_r;
	if(nrhs>=3){

		stride_in = (double*) mxGetData(STRIDE_IN);
		stride_c = (size_t)stride_in[0];
		stride_r = (size_t)stride_in[1];
	}
	else{
		stride_c = 1;
		stride_r = 1;
	}



	int padding=0;
	if(nrhs==4)
		padding = mxGetScalar(PADDING);

	float *img_in;
	float *img_out;
	img_in = (float*)mxGetData(IMG_IN);


	double* filter_size;
	filter_size = (double*) mxGetData(KSIZE_IN);
	size_t k_height, k_width;
	k_height = (size_t)filter_size[0];
	k_width = (size_t)filter_size[1];

	




	size_t img_height =mxGetDimensions(IMG_IN)[0];
	size_t img_width = mxGetDimensions(IMG_IN)[1];
	size_t img_channel = mxGetDimensions(IMG_IN)[2];

	//mexPrintf("height %d width %d channel %d\n", img_height, img_width, img_channel);
	//mexPrintf("filter height %d filter width %d \n", k_height, k_width);
	

	size_t total_size = ((img_height+2*padding-k_height)/stride_c+1)*((img_width+2*padding-k_width)/stride_r+1);
	size_t col_size = k_height*k_width*img_channel;
	int thread_num  = total_size*img_channel;
	size_t height_out = (img_height+2*padding-k_height)/stride_c+1;
	size_t width_out = (img_width+2*padding-k_width)/stride_r+1;

	//mexPrintf("Total %d columns \n", total_size);
	
	IMG_OUT = mxCreateNumericMatrix(total_size, col_size, mxSINGLE_CLASS, mxREAL);
	img_out = (float*)mxGetData(IMG_OUT);

	/* Call matlab function permute to change to row-major*/

	mxArray* input_array[2], *output_array[1];
	input_array[0] = mxDuplicateArray(IMG_IN);

	mxArray* DIM_ORDER = mxCreateNumericMatrix(1,3,mxDOUBLE_CLASS, mxREAL);
	double* ptr = (double*)mxGetData(DIM_ORDER);
	ptr[0]=2; ptr[1]=1; ptr[2]=3;

	input_array[1] = DIM_ORDER;

	mexCallMATLAB(1, output_array, 2, input_array, "permute");

	mxArray* RM_IMG_IN = output_array[0];

	img_in = (float*)mxGetData(RM_IMG_IN);


	/* Start CUDA PROCESSING*/
	float *d_img;
	float *d_col;
	size_t n_pixels = img_height*img_width*img_channel;
	gpuErrchk(cudaMalloc(&d_img, n_pixels*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_col, col_size*total_size*sizeof(float)));


	/* Copy data to GPU mem */
	//copy data
	gpuErrchk(cudaMemcpy(d_img, img_in, n_pixels*sizeof(float), cudaMemcpyHostToDevice));

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	im2col_gpu_kernel<<<32*numSMs, THREAD_PER_BLOCK>>>(thread_num, d_img,
		img_height, img_width, k_height, padding, stride_c,
		height_out, width_out,
		d_col);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	/*copy result back to cpu mem */
	gpuErrchk(cudaMemcpy(img_out, d_col, col_size*total_size*sizeof(float), cudaMemcpyDeviceToHost));

	mxSetN(DIM_ORDER, 2);
	input_array[0] = IMG_OUT;
	input_array[1] = DIM_ORDER;
	mexCallMATLAB(1, output_array, 2, input_array, "permute");

	mxDestroyArray(IMG_OUT);
	IMG_OUT = output_array[0];

	gpuErrchk(cudaFree(d_img));
	gpuErrchk(cudaFree(d_col));
	return;
}
