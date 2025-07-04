#include <assert.h>
#include <cstdio>
#include <stdio.h>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

#define ELEMENTS_PER_THREAD 8

#define USE_FLOAT4 1
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
                              int hA) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * ELEMENTS_PER_THREAD * BLOCK_SIZE * by;
  int aEnd = aBegin + wA - 1;
  int aStep = BLOCK_SIZE;
  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * wB;

  float Csub[ELEMENTS_PER_THREAD];
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    Csub[i] = 0.0f;
  }

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

    __shared__ float As[ELEMENTS_PER_THREAD * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    Bs[ty][tx] = B[b + wB * ty + tx];

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
      int global_row =
          ELEMENTS_PER_THREAD * BLOCK_SIZE * by + (ty + i * BLOCK_SIZE);
      if (global_row < hA) {
        As[ty + i * BLOCK_SIZE][tx] = A[a + wA * (ty + i * BLOCK_SIZE) + tx];
      } else {
        As[ty + i * BLOCK_SIZE][tx] = 0.0f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
#if USE_FLOAT4
      float4 b_vec = make_float4(Bs[k][tx], Bs[k][tx], Bs[k][tx], Bs[k][tx]);
      for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
        if (i + 3 < ELEMENTS_PER_THREAD) {
          float4 a_vec = make_float4(As[ty + i * BLOCK_SIZE][k],
                                     As[ty + (i + 1) * BLOCK_SIZE][k],
                                     As[ty + (i + 2) * BLOCK_SIZE][k],
                                     As[ty + (i + 3) * BLOCK_SIZE][k]);

          float4 result = make_float4(a_vec.x * b_vec.x, a_vec.y * b_vec.y,
                                      a_vec.z * b_vec.z, a_vec.w * b_vec.w);

          Csub[i] += result.x;
          Csub[i + 1] += result.y;
          Csub[i + 2] += result.z;
          Csub[i + 3] += result.w;
        } else {
          for (int j = i; j < ELEMENTS_PER_THREAD; j++) {
            Csub[j] += As[ty + j * BLOCK_SIZE][k] * Bs[k][tx];
          }
          break;
        }
      }
#else
      for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        Csub[i] += As[ty + i * BLOCK_SIZE][k] * Bs[k][tx];
      }
#endif
    }

    __syncthreads();
  }

  int c = wB * ELEMENTS_PER_THREAD * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int global_row =
        ELEMENTS_PER_THREAD * BLOCK_SIZE * by + (ty + i * BLOCK_SIZE);
    if (global_row < hA) {
      C[c + wB * (ty + i * BLOCK_SIZE) + tx] = Csub[i];
    }
  }
}

#define MAX_VAL 1.0f
#define MIN_VAL 0.0f

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void GradientInit(float *data, int width, int height, int multiplier = 1) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float gradient_value = (float)col / (float)(width - 1);
      data[row * width + col] = gradient_value * multiplier;
    }
  }
}

void NormalizationInit(float *data, int width, int height, int original_width) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      if (row == col && row < width && row < height) {
        data[row * width + col] = (float)original_width;
      } else {
        data[row * width + col] = 0.0f;
      }
    }
  }
}

void MatrixMultiplyCPU(float *C, const float *A, const float *B, int hA, int wA, int wB) {
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

  int multiplier = 2;
  GradientInit(h_A, dimsA.x, dimsA.y);
  ConstantInit(h_B, size_B, multiplier);

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

  dim3 grid(dimsB.x / threads.x,
            (dimsA.y + ELEMENTS_PER_THREAD * threads.y - 1) /
                (ELEMENTS_PER_THREAD * threads.y),
            1);

  printf("Computing result using CUDA Kernel...\n");

  if (block_size == 16) {
    MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                    dimsB.x, dimsA.y);
  } else {
    MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                    dimsB.x, dimsA.y);
  }

  printf("done\n");
  checkCudaErrors(cudaStreamSynchronize(stream));

  checkCudaErrors(cudaEventRecord(start, stream));

  int nIter = 1;

  for (int j = 0; j < nIter; j++) {
    if (block_size == 16) {
      MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                      dimsB.x, dimsA.y);
    } else {
      MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x,
                                                      dimsB.x, dimsA.y);
    }
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

  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

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
  bool correct = true;

  double eps = 1.e-4;
  for (int i = 0; i < dimsC.y; i++) {
    for (int j = 0; j < dimsC.x; j++) {
      float val = h_C[i * dimsC.x + j];
      float ref = h_C_ref[i * dimsC.x + j];
      if (fabs(val - ref) > eps * fabs(ref)) {
        if (fabs(ref) > eps) {
          printf("Error at (%d, %d): GPU result = %f, CPU result = %f\n",
                 i, j, val, ref);
          correct = false;
          break;
        }
      }
    }
    if (!correct)
      break;
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

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
    printf("      -blocksize=n (n = 16 or 32, default is 16)\n");
    printf(
        "      -elements_per_thread=n (n = 1, 2, 4, 8, etc., default is 4)\n");

    exit(EXIT_SUCCESS);
  }

  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 16;

  if (checkCmdLineFlag(argc, (const char **)argv, "blocksize")) {
    block_size = getCmdLineArgumentInt(argc, (const char **)argv, "blocksize");
    if (block_size != 16 && block_size != 32) {
      printf("Error: block size must be 16 or 32.\n");
      exit(EXIT_FAILURE);
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "elements_per_thread")) {
    int elements_per_thread =
        getCmdLineArgumentInt(argc, (const char **)argv, "elements_per_thread");
    if (elements_per_thread < 1 || elements_per_thread > 8) {
      printf("Error: elements per thread must be between 1 and 8.\n");
      exit(EXIT_FAILURE);
    }
    if (elements_per_thread != ELEMENTS_PER_THREAD) {
      printf("Warning: changing ELEMENTS_PER_THREAD to %d.\n",
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

  printf("ELEMENTS PER THREAD: %d\n", ELEMENTS_PER_THREAD);

  printf("Grid SIZE: (%d, %d)\n", (int)(dimsB.x / block_size),
         (int)((dimsA.y + ELEMENTS_PER_THREAD * block_size - 1) /
               (ELEMENTS_PER_THREAD * block_size)));
  printf("BLOCK SIZE: (%d, %d)\n", block_size, block_size);
  printf("EACH BLOCK PROCESS %d ROWS\n",
         ELEMENTS_PER_THREAD * block_size);
  printf("ALL BLOCKS IN Y : %d, X : %d\n",
         (int)((dimsA.y + ELEMENTS_PER_THREAD * block_size - 1) /
               (ELEMENTS_PER_THREAD * block_size)),
         (int)((dimsA.y + ELEMENTS_PER_THREAD * block_size - 1) /
               (ELEMENTS_PER_THREAD * block_size)) *
             ELEMENTS_PER_THREAD * block_size);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
