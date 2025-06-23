#include "matrix_mul.hh"

__global__ void matMulGPU_naive(matrix_gpu C, const matrix_gpu A,
                                const matrix_gpu B) {
  // TODO: implement the naive version of matmul here
  int i=blockIdx.y * blockDim.y + threadIdx.y;
  int j=blockIdx.x * blockDim.x + threadIdx.x;
  if (i<C.rows() && j<C.cols())

      for (int k=0; k<A.cols(); k++){
        C(i,j)=C(i,j)+A(i,k)*B(k,j);
      }
}


__global__ void matMulGPU_shared(matrix_gpu C, const matrix_gpu A, const matrix_gpu B) {
  // Block row and column
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // Shared memory tiles
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0;

  // Load tile from A and B into shared memory
  for (int t = 0; t < (A.cols() + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
    if (row < A.rows() && t * BLOCK_SIZE + threadIdx.x < A.cols())
      As[threadIdx.y][threadIdx.x] = A(row, t * BLOCK_SIZE + threadIdx.x);
    else
      As[threadIdx.y][threadIdx.x] = 0;

    if (t * BLOCK_SIZE + threadIdx.y < B.rows() && col < B.cols())
      Bs[threadIdx.y][threadIdx.x] = B(t * BLOCK_SIZE + threadIdx.y, col);
    else
      Bs[threadIdx.y][threadIdx.x] = 0;

    __syncthreads(); 

    for (int k = 0; k < BLOCK_SIZE; k++)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();  
  }

  if (row < C.rows() && col < C.cols())
    C(row, col) = sum;
}