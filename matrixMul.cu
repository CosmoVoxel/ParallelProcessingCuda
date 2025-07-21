#include <__clang_cuda_builtin_vars.h>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <stdio.h>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_functions.h"
#include "matrixMulConfig.h"

__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
                              int hA) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * elements_per_thread_x * block_size * by;
  int aEnd = aBegin + wA - 1;
  int aStep = block_size;
  int bBegin = block_size * bx * elements_per_thread_y;
  int bStep = block_size * wB;

  const int total_elements = elements_per_thread_x * elements_per_thread_y;

  float Csub[total_elements] = {0.0f};

  __shared__ float As[elements_per_thread_x * block_size][block_size];
  __shared__ float Bs[block_size][elements_per_thread_y * block_size];

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

#pragma unroll
    for (int i = 0; i < elements_per_thread_x; i++) {
      int row = ty + i * block_size;
      int col = tx;
      As[row][tx] = A[a + wA * row + col];
    }
#pragma unroll
    for (int i = 0; i < elements_per_thread_y; i++) {
      int col = tx + i * block_size;
      Bs[ty][col] = B[b + (wB * ty) + col];
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < block_size; ++k) {
      for (int i = 0; i < elements_per_thread_x; i++) {
        for (int j = 0; j < elements_per_thread_y; j++) {
          Csub[j + i * elements_per_thread_y] +=
              As[ty + i * block_size][k] * Bs[k][tx + j * block_size];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < elements_per_thread_x; i++) {
    for (int j = 0; j < elements_per_thread_y; j++) {
      int row = (by * block_size * elements_per_thread_x) + ty + i * block_size;
      int col = (bx * block_size * elements_per_thread_y) + tx + j * block_size;
      if (row < hA && col < wB) {
        C[row * wB + col] = Csub[j + i * elements_per_thread_y];
      }
    }
  }
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void RandomInit(float *data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void GradientInit(float *data, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float gradient_value = (float)col / (float)(width - 1);
      data[row * width + col] = gradient_value;
    }
  }
}

void MatrixMultiplyCPU(float *C, const float *A, const float *B, int hA, int wA,
                       int wB) {
  for (int row = 0; row < hA; ++row) {
    for (int col = 0; col < wB; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < wA; ++k) {
        sum += A[row * wA + k] * B[k * wB + col];
      }
      C[row * wB + col] = sum;
    }
  }
}

int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {

  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  RandomInit(h_A, size_A);
  RandomInit(h_B, size_B);

  int min_dim = (dimsB.x < dimsB.y) ? dimsB.x : dimsB.y;
  for (int i = 0; i < min_dim; i++) {
    h_B[i * dimsB.x + i] = 1.0f;
  }

  float *d_A, *d_B, *d_C;

  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  dim3 threads(block_size, block_size);

  int grid_x = (dimsB.x + threads.x * elements_per_thread_y - 1) /
               (threads.x * elements_per_thread_y);
  int grid_y = (dimsA.y + threads.y * elements_per_thread_x - 1) /
               (threads.y * elements_per_thread_x);

  dim3 grid(grid_x, grid_y, 1);

  printf("Computing result using CUDA Kernel...\n");

  MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                    dimsB.x, dimsA.y);

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  checkCudaErrors(cudaEventRecord(start, stream));

  int nIter = 30;

  for (int j = 0; j < nIter; j++) {
      MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                      dimsB.x, dimsA.y);
      MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                      dimsB.x, dimsA.y);
  }

  checkCudaErrors(cudaEventRecord(stop, stream));

  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
         " WorkgroupSize= %u threads/block\n",
         gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  
  printf("Grid SIZE: (%d, %d)\n", grid.x, grid.y);

  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  bool correct = true;
  if (verify) {
    printf("Checking computed result for correctness: ");
    printf("Running CPU reference...\n");
    float *h_C_ref;
    checkCudaErrors(cudaMallocHost(&h_C_ref, mem_size_C));
    if (h_C_ref == NULL) {
      fprintf(stderr, "Failed to allocate host matrix C reference!\n");
      exit(EXIT_FAILURE);
    }
    MatrixMultiplyCPU(h_C_ref, h_A, h_B, dimsA.y, dimsA.x, dimsB.x);
    printf("done\n");
    printf("Comparing results...\n");
    double eps = 1.e-4;
    int error_count = 0;
    for (int i = 0; i < dimsC.y; i++) {
      for (int j = 0; j < dimsC.x; j++) {
        float val = h_C[i * dimsC.x + j];
        float ref = h_C_ref[i * dimsC.x + j];
        if (fabs(val - ref) > eps * fabs(ref)) {
          if (fabs(ref) > eps) {
            printf("Error at (%d, %d): GPU result = %f, CPU result = %f\n", i,
                   j, val, ref);
            error_count++;
            correct = false;
          }
        }
      }
      if (!correct)
        break;
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    if (error_count > 0) {
      printf("Total errors: %d\n", error_count);
    }
  }

  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  printf("\nNOTE: The CUDA Samples are not meant for performance "
         "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices"
           " must be equal.\n");
    printf("-blocksize=n (n = 16 or 32, default is 16)\n");
    exit(EXIT_SUCCESS);
  }

  int dev = findCudaDevice(argc, (const char **)argv);

  if (checkCmdLineFlag(argc, (const char **)argv, "elements_per_thread")) {
    int elements_per_thread =
        getCmdLineArgumentInt(argc, (const char **)argv, "elements_per_thread");
    if (elements_per_thread < 1 || elements_per_thread > 8) {
      printf("Error: elements per thread must be between 1 and 8.\n");
      exit(EXIT_FAILURE);
    }
    if (elements_per_thread != elements_per_thread_x) {
      printf("Warning: changing elements_per_thread_x to %d.\n",
             elements_per_thread);
    }
  }

  int wA = 50 * block_size;
  int both = 50 * block_size;
  int hB = 50 * block_size;

  wA *= 1;
  both *= 1;
  hB *= 1;

  dim3 dimsA(wA, both, 1);
  dim3 dimsB(both, hB, 1);

  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  printf("X ELEMENTS PER THREAD: %d\n", elements_per_thread_x);
  printf("Y ELEMENTS PER THREAD: %d\n", elements_per_thread_y);

  printf("BLOCK SIZE: (%d, %d)\n", block_size, block_size);
  printf("EACH THREAD WILL FETCH %d ROWS\n",
         elements_per_thread_x * block_size);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
