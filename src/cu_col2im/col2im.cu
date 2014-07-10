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

__global__ void col2im_gpu_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}


#define THREAD_PER_BLOCK 256


#define IMG_OUT plhs[0]

#define COL_IN prhs[0]
#define IMG_SIZE prhs[1]
#define BLOCK_SIZE prhs[2]
#define STRIDE_SIZE prhs[3]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs<3)
		mexErrMsgTxt("Not enough inputs");

	double* block_size = (double *) mxGetData(BLOCK_SIZE);
	size_t block_height = (size_t) block_size[0];
	size_t block_width = (size_t) block_size[1];
	size_t block_depth = (size_t) block_size[2];

	double* img_size = (double*) mxGetData(IMG_SIZE);
	size_t img_height = (size_t) img_size[0];
	size_t img_width = (size_t) img_size[1];
	size_t img_channel = (size_t) img_size[2];
	size_t img_size_int[3] = {img_height, img_width, img_channel};
	

	double* stride_in;
	size_t stride_c, stride_r, stride_d;
	if(nrhs==4){

		stride_in = (double*) mxGetData(STRIDE_SIZE);
		stride_c = (size_t)stride_in[0];
		stride_r = (size_t)stride_in[1];
		stride_d = (size_t)stride_in[2];
	}
	else{
		stride_c = 1;
		stride_r = 1;
		stride_d = 1;
	}
	size_t real_c_size = img_height/stride_c;
	size_t real_r_size = img_width/stride_r;
	size_t real_d_size = img_channel/stride_d;
	size_t out_mem_size = real_c_size*real_r_size*real_d_size*sizeof(float);


	float *col_in;
	float *img_out;
	col_in = (float*)mxGetData(COL_IN);

	size_t col_height =mxGetDimensions(COL_IN)[0];
	size_t col_width = mxGetDimensions(COL_IN)[1];

	mxAssert(col_height==block_depth*block_height*block_width, "Column size incorrect.");

	IMG_OUT = mxCreateNumericArray(3, img_size_int, mxSINGLE_CLASS, mxREAL);

	img_out = (float*) mxGetData(IMG_OUT);

	float* d_col;
	float *d_img;

	gpuErrchk(cudaMalloc(&d_col, col_height*col_width*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_img, out_mem_size));

	size_t thread_size = col_width*img_channel/(block_height*block_width);


	/* Call matlab function permute to change to row-major*/

	mxArray* input_array[2], *output_array[1];
	input_array[0] = mxDuplicateArray(COL_IN);

	mxArray* DIM_ORDER = mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS, mxREAL);
	double* ptr = (double*)mxGetData(DIM_ORDER);
	ptr[0]=2; ptr[1]=1;

	input_array[1] = DIM_ORDER;

	mexCallMATLAB(1, output_array, 2, input_array, "permute");

	mxArray* RM_COL_IN = output_array[0];

	col_in = (float*)mxGetData(RM_COL_IN);

	int numSMs;
	gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

	/* Copy mem to GPU */
	gpuErrchk(cudaMemcpy(d_col, col_in, col_height*col_width*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_img, 0, out_mem_size));



	col2im_gpu_kernel<<<32*numSMs, THREAD_PER_BLOCK>>>(thread_size, 
		d_col,
		img_height, img_width, img_channel, block_height,
	0, stride_c, img_height, img_width,
    d_img);

	gpuErrchk(cudaDeviceSynchronize());

	/* copy mem back to gpu */
	gpuErrchk(cudaMemcpy(img_out, d_img, out_mem_size, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_img));
	gpuErrchk(cudaFree(d_col));
	return;

}