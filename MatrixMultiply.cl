// OpenCL Kernel for matrix Multiply
__kernel void Matrix(__global int* C, __global int* A, __global int* B, int size){
  
   // 2D Thread ID
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   // value stores the element that is computed by the thread
   int value = 0;
   
   for (int k = 0; k < size; ++k){
      value += A[ty * size + k] * B[k * size + tx];
   }

   C[ty * size + tx] = value;
}
