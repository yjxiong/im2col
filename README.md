im2col
======

Transfer image to columns to support all kinds of sliding window operations

## Usage

1. generate the VS solution using cmake.
2. Build with MSVS and the output binary will be in matlab/bin
3. Run `matlab/test_im2col.m` or `matlab/test_col2im.m` to debug or test.

## About the test

- Test scripts have self-explaning paramters;
- The test scripts will run the cuda version mex function;
- They also runs a cpu implementation of im2col/col2im;
- Error of the function are calculated by comparing these two results. `err=0` means the cuda kernel is running correctly.
