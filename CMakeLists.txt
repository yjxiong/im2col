cmake_minimum_required (VERSION 2.8)
project (im2col)

if(MSVC)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#set(THREADS_USE_PTHREADS_WIN32 TRUE)

  string(REGEX REPLACE "/Z[iI7]" ""
         CMAKE_CXX_FLAGS_RELEASE
         "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Z7")
endif()




find_package (CUDA REQUIRED)

find_package (MATLAB REQUIRED)

# nvcc flags                                        
#list(APPEND CUDA_NVCC_FLAGS )

if (MSVC)
include_directories (
  ${CUDA_INCLUDE_DIRS}
  ${MATLAB_INCLUDE_DIR}
  ${MATLAB_GPU_INCLUDE_DIR}
  include
)
else()
include_directories (
  ${CUDA_INCLUDE_DIRS}
  ${MATLAB_INCLUDE_DIR}
  include
)
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30;-Xcompiler \"/wd 4819\"")
CUDA_ADD_LIBRARY (cu_im2col SHARED
  src/cu_im2col/im2col.cu
  src/cu_im2col/matlab.def
)

CUDA_ADD_LIBRARY (cu_col2im SHARED
  src/cu_col2im/col2im.cu
  src/cu_col2im/matlab.def
)

set_target_properties (cu_im2col
  PROPERTIES
  PREFIX ""
  SUFFIX ".mexw64"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ../matlab/bin
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ../matlab/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ../matlab/bin
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ../matlab/bin
)
set_target_properties (cu_col2im
  PROPERTIES
  PREFIX ""
  SUFFIX ".mexw64"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ../matlab/bin
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ../matlab/bin
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ../matlab/bin
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ../matlab/bin
)



TARGET_LINK_LIBRARIES (cu_im2col
  ${MATLAB_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)
TARGET_LINK_LIBRARIES (cu_col2im
  ${MATLAB_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)


CUDA_BUILD_CLEAN_TARGET()
