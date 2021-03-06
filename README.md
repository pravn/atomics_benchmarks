# atomics_benchmarks
Benchmark to calculate atomics bandwidth in different GPU architectures. It should give a *rough* measure of shared memory atomics performance in various GPU architectures. We emphasize 'rough' because both global and shared memory atomics operations are involved. Nevertheless, for the operation in question - that of computing a histogram - this is a plausible approach both in terms of ease of programmability and, we think, in elucidating the use if atomics as the feature of choice for this task. It is not unlikely that there might be ample room for optimization opportunities in the implementation. 

   
It computes a histogram of values that fall into a number of bins. 
The worst case performance scenario occurs when all the data falls into
a single bin (bin 0). This is turned on (which is the default) by setting the WORST_CASE flag (see in 'building' below. The other, more favorable scenario (remove the WORST_CASE flag in cmake) has the data being generated such that the bins are selected randomly from 0 to NUM_BINS-1. 

**GPU algorithm**:
The calculation is performed in the shmem_atomics_reducer kernel. We accumulate histogram values into bin variables in shared memory using atomicAdd (shared memory atomics). These are then written to global memory in coalesced fashion.


#Building
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


#Running
The executable will build in <build-dir>/histogram.
Run as follows:
cd <build-dir>/histogram <br>
./histogram <br>

Example output: See example_output.txt


