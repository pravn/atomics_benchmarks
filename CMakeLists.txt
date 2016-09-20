cmake_minimum_required(VERSION 2.8)
Project(cuda_experiments)
SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-7.0)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(include)

#sm_30 - Kepler
#sm_35 - Kepler 
#sm_52 - Maxwell
#sm_62 - Pascal, DrivePX2 - works only for CUDA-8

SET(CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30;
-gencode arch=compute_35,code=sm_35;
-gencode arch=compute_52,code=sm_52;
#-gencode arch=compute_62,code=sm_62
-std=c++11")



ADD_SUBDIRECTORY(histogram)
