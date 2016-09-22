#include <iostream>
#include <cuda.h>
#include <iomanip>
#include <Timer.h>
#include <omp.h>

//histogram with N bins in several blocks 
//compute histogram using shared memory atomics
//do a reduction (atomicAdd) in shared memory
//then do a coalesced write to global memory (atomicAdd)

const long int NUM_BLOCKS=256;
#define NUM_BINS 32
#define NUM_THREADS_PER_BLOCK 256


__global__ void shmem_atomics_reducer(int *data, int *count){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int block_reduced[NUM_THREADS_PER_BLOCK];
  block_reduced[threadIdx.x] = 0;

  __syncthreads();

    atomicAdd(&block_reduced[data[tid]],1);
  __syncthreads();

  for(int i=threadIdx.x; i<NUM_BINS; i+=NUM_BINS)
    atomicAdd(&count[i],block_reduced[i]);
  
}
	  

void run_atomics_reducer(int *h_data){
  int *d_data;
  int *h_result_atomics;
  int *d_result_atomics;
  int *h_result;

  cudaMalloc((void **) &d_data, NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int));
  cudaMemcpy(d_data, h_data, NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

  h_result = new int[NUM_BINS];
  memset(h_result, 0, NUM_BINS*sizeof(int));

  cudaMalloc((void **) &d_result_atomics, NUM_BINS*sizeof(int));
  cudaMemset(d_result_atomics, 0, NUM_BINS*sizeof(int));

  CUDATimer atomics_timer;
  double gpu_time = 0;
  int niter = 10;


  for(int i=0; i<niter; i++){
	  cudaMemset(d_result_atomics, 0, NUM_BINS*sizeof(int));
	  atomics_timer.startTimer();
	  shmem_atomics_reducer<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_atomics);
	  atomics_timer.stopTimer();
	  gpu_time  += atomics_timer.getElapsedTime();
  }


CPUTimer cpu_timer;
double cpu_time = 0;

 int nproc = omp_get_num_procs();

 omp_set_num_threads(nproc);

 std::cout << "Number of CPU cores for use in OpenMP " << nproc << std::endl;

 for(int iter=0; iter<niter; iter++){
   memset(h_result, 0, NUM_BINS*sizeof(int));
   cpu_timer.startTimer();
#pragma omp parallel for   
   for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
     for(int j=0; j<NUM_BINS; j++){
       if(h_data[i]==j){
	 #pragma omp atomic
	 h_result[j]++;
       }
     }
   }
   cpu_timer.stopTimer();
   cpu_time  += cpu_timer.getElapsedTime();
 }


  h_result_atomics = new int[NUM_BINS];
  cudaMemcpy(h_result_atomics, d_result_atomics, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "=======================================" << std::endl;
  std::cout << "Computed average atomics bandwidth for " << niter << " iterations " << std::endl;
  std::cout << "Average GPU atomics time in milliseconds " << gpu_time/niter << std::endl;
  std::cout << "Average CPU atomics time in milliseconds " << cpu_time/niter << std::endl;

  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;

  std::cout << "Megabytes of data " << mbytes << std::endl;

   float gpu_bandwidth = mbytes/gpu_time*niter*1e3;
   float cpu_bandwidth = mbytes/cpu_time*niter*1e3;

   #ifdef WORST_CASE
   std::cout << "Running worst case scenario where all data falls into a single bin " << std::endl;
   #else 
   std::cout << "Running for case where data is distributed randomly into " << NUM_BINS << " bins " << std::endl;
   #endif 

  std::cout << "GPU Atomics bandwidth in MB/s " << gpu_bandwidth << std::endl;
  std::cout << "CPU Atomics bandwidth in MB/s " << cpu_bandwidth << std::endl;
  


  std::cout << "Validation: " << std::endl;
  std::cout << std::setw(4) << "BIN#" 
	    << std::setw(7) << "HOST"
	    << std::setw(9) << "DEVICE" << std::endl;

  for(int i=0; i<NUM_BINS; i++){
    std::cout <<  std::setw(4) << i << " " <<  std::setw(6) << h_result[i] 
	      << " " << std::setw(7) << h_result_atomics[i] << std::endl;
    }

  
  cudaFree(d_data);
  delete[] h_result_atomics;
  cudaFree(d_result_atomics);
  delete[] h_result;

}
  
  


int main()
{
  int *h_data; 
  h_data = new int[NUM_THREADS_PER_BLOCK*NUM_BLOCKS];

  std::cout << "Data Size " << NUM_THREADS_PER_BLOCK * NUM_BLOCKS << std::endl;

  for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
#ifdef WORST_CASE //worst case scenario when all pixels fall into a single bin
    	   h_data[i] = 0; 
#else
            h_data[i] = (NUM_BINS) * ((float) rand()/RAND_MAX);
#endif
  }

  run_atomics_reducer(h_data);



  //cleanup
  delete[] h_data;

}
