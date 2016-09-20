# atomics_benchmarks
Benchmark to calculate atomics bandwidth in different GPU architectures. 
It computes a histogram of values that fall into a number of bins. 
The worst case performance scenario occurs when all the data falls into
a single bin (bin 0). This is turned on (which is the default) by setting the WORST_CASE flag (see in 'building' below. The other, more favorable scenario (remove the WORST_CASE flag in cmake) has the data being generated such that the bins are selected randomly from 0 to NUM_BINS-1. 

GPU algorithm:
The calculation is performed in the shmem_atomics_reducer kernel. We accumulate histogram values into bin variables in shared memory using atomicAdd. These are then written to global memory in coalesced fashion.


Building:
Create a build directory and configure with cmake.

mkdir bld <br>
cmake \<path/to/src>

We might have to modify paths to CUDA toolkit directory in CMakeLists. The gpu architecture flags will also need modification depending on the hardware it is run on. 

A caveat noticed is that the DrivePX2 architecture needs CUDA 8.0 to be installed. The CUDA toolkit specification will have to be changed accordingly.

Building for the worst case. This is set in histogram/CMakeLists by passing the flag -DWORST_CASE. It is turned on by default. It should be unset (removed) in histogram/CMakeLists if we want to run for the more favorable scenario wherein data is distributed randomly into all the bins. 

We also set the flags defined on top of the histogram.cu file. 

The following flags are of interest: <br>
NUM_BLOCKS <br>
NUM_BINS   <br> 
NUM_THREADS_PER_BLOCK <br>


Running:
The executable will build in <build-dir>/histogram.
Run as follows:
cd <build-dir>/histogram
./histogram

Example output:
Data Size 65536
======================================
atomics time in milliseconds 0.025824
Megabytes of data 0.262144
Running worst case scenario where all data falls into a single bin 
Atomics bandwidth in MB/s 10151.2
Validation: 
BIN#   HOST   DEVICE
   0  65536   65536
   1      0       0
   2      0       0
   3      0       0
   4      0       0
   5      0       0
   6      0       0
   7      0       0
   8      0       0
   9      0       0
  10      0       0
  11      0       0
  12      0       0
  13      0       0
  14      0       0
  15      0       0
  16      0       0
  17      0       0
  18      0       0
  19      0       0
  20      0       0
  21      0       0
  22      0       0
  23      0       0
  24      0       0
  25      0       0
  26      0       0
  27      0       0
  28      0       0
  29      0       0
  30      0       0
  31      0       0



 



