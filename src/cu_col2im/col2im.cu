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


HEMI_KERNEL(d_col2im)(float* d_col,  int col_c_size, int col_r_size,
					  int thread_size,
					  int img_c_size, int img_r_size, int img_channel,
					  int block_c_size, int block_r_size,
					  int stride_c, int stride_r, int stride_d,	
					  int real_c_size, int real_r_size, int real_d_size,
					  float* d_output_img)
{
	HEMI_GRID_STRIDE_LOOP(idx, thread_size){
		/* For col_r_size columns, we recompile them to an image */
		int index = idx%col_r_size;
		int ch = idx/col_r_size;
		int r_id = index%real_c_size;
		int c_id_t = index/real_c_size;
		int c_id = c_id_t%real_r_size;
		int d_id = c_id_t/real_r_size;
		int depth_step = img_c_size*img_r_size;
		int block_depth_step = block_c_size*block_r_size;

		int base = r_id*stride_c+
			c_id*stride_r*img_c_size+
			d_id*stride_d*img_c_size*img_r_size+ch*depth_step;

		int col_base = index*col_c_size+ch*block_depth_step;



		for (int k_c=0; k_c<block_r_size; k_c++)
			for (int k_r=0; k_r<block_c_size; k_r++){
				d_output_img[base+k_r+k_c*img_c_size] +=
					d_col[col_base+k_r+k_c*block_c_size];
			}

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

	int numSMs;
	gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

	/* Copy mem to GPU */
	gpuErrchk(cudaMemcpy(d_col, col_in, col_height*col_width*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_img, 0, out_mem_size));


	/*	Call Cuda Kernel via Hemi	*/
	HEMI_KERNEL_LAUNCH(d_col2im, 32*numSMs, THREAD_PER_BLOCK, 0, 0, 
		d_col,
		col_height, col_width, 
		thread_size,
		img_height, img_width, img_channel,
		block_height, block_width,
		stride_c,stride_r,stride_d,
		real_c_size, real_r_size, real_d_size,
		d_img
		);

	gpuErrchk(cudaDeviceSynchronize());

	/* copy mem back to gpu */
	gpuErrchk(cudaMemcpy(img_out, d_img, out_mem_size, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_img));
	gpuErrchk(cudaFree(d_col));
	return;

}