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



HEMI_KERNEL(d_im2col)(float* d_image,  int img_c_size, int img_r_size,
					  int thread_num, int total_length, 
					  int ksize_c, int ksize_r, int channels, 
					  int stride_c, int stride_r,		  
					  int padding_c, int padding_r, float* d_output)
{
	int out_c_size, out_r_size, out_row, out_col, ch;
	int col_length = ksize_c*ksize_r*channels;
	int ksize = ksize_c*ksize_r;
	int img_channel_size = img_c_size*img_r_size;
	out_c_size = (img_c_size-ksize_c+2*padding_c)/stride_c+1;
	out_r_size = (img_r_size-ksize_r+2*padding_r)/stride_r+1;
	HEMI_GRID_STRIDE_LOOP(idx, thread_num){

		//transform the image data to columns
		
		int index = idx;
		out_row = index%out_c_size;
		index/=out_c_size;
		out_col = index%out_r_size;
		ch = index/out_r_size;
		int col_base = col_length*(idx%total_length)+ksize*ch;
		int img_base = img_channel_size*ch;

		int ori_c_zero= out_col*stride_c-padding_c;
		int ori_r_zero= out_row*stride_r-padding_r;

			for (int k_c=0; k_c<ksize_r; k_c++)
				for (int k_r=0; k_r<ksize_c; k_r++){
					int ori_c = ori_c_zero+k_c;
					int ori_r = ori_r_zero+k_r;
					d_output[col_base+ k_c*ksize_c+k_r]=(ori_c>=0&&ori_c<img_c_size&&ori_r>=0&&ori_r<img_r_size)?
						d_image[img_base+ori_c*img_c_size+ori_r]
						//ch*ksize+ k_c*ksize_c+k_r
					:0;
					//d_output[idx] = col_base;
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

	//mexPrintf("Total %d columns \n", total_size);
	
	IMG_OUT = mxCreateNumericMatrix(col_size, total_size, mxSINGLE_CLASS, mxREAL);
	img_out = (float*)mxGetData(IMG_OUT);


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

	
	/* launch kernel */
	HEMI_KERNEL_LAUNCH(d_im2col, 32*numSMs, THREAD_PER_BLOCK, 0, 0, 
		d_img, img_height, img_width, 
		thread_num, total_size,
		k_height, k_width, img_channel,
		stride_c,stride_r,
		padding, padding, 
		d_col);
	gpuErrchk(cudaDeviceSynchronize());
	/*copy result back to cpu mem */
	gpuErrchk(cudaMemcpy(img_out, d_col, col_size*total_size*sizeof(float), cudaMemcpyDeviceToHost));



	gpuErrchk(cudaFree(d_img));
	gpuErrchk(cudaFree(d_col));
	return;
}
