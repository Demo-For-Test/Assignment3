// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#include "cuda_kernels.h"
#include <cmath>

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int SHARED_W = TILE_W + 2; // 18x18 tile with 1-pixel halo
constexpr int SHARED_H = TILE_H + 2;

__global__ void RgbToGrayscaleKernel(const uint8_t* __restrict__ d_rgb,
                                    uint8_t* __restrict__ d_gray,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx_gray = y * width + x;
  int idx_rgb = idx_gray * 3;

  float r = static_cast<float>(d_rgb[idx_rgb]);
  float g = static_cast<float>(d_rgb[idx_rgb + 1]);
  float b = static_cast<float>(d_rgb[idx_rgb + 2]);

  // Perceptual luminance calculation (ITU-R BT.601)
  float gray = 0.299f * r + 0.587f * g + 0.114f * b;
  d_gray[idx_gray] = static_cast<uint8_t>(fminf(fmaxf(gray, 0.0f), 255.0f));
}

__global__ void GaussianBlurSharedMemKernel(const uint8_t* __restrict__ d_input,
                                           uint8_t* __restrict__ d_output,
                                           int width, int height) {
  __shared__ uint8_t tile[SHARED_H][SHARED_W];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  // Load interior of shared tile
  int clamped_gx = min(max(gx, 0), width - 1);
  int clamped_gy = min(max(gy, 0), height - 1);
  tile[ty + 1][tx + 1] = d_input[clamped_gy * width + clamped_gx];

  // Load halo boundaries
  if (tx == 0) {
    int hx = min(max(gx - 1, 0), width - 1);
    tile[ty + 1][0] = d_input[clamped_gy * width + hx];
  }
  if (tx == blockDim.x - 1 || gx == width - 1) {
    int hx = min(max(gx + 1, 0), width - 1);
    tile[ty + 1][tx + 2] = d_input[clamped_gy * width + hx];
  }
  if (ty == 0) {
    int hy = min(max(gy - 1, 0), height - 1);
    tile[0][tx + 1] = d_input[hy * width + clamped_gx];
  }
  if (ty == blockDim.y - 1 || gy == height - 1) {
    int hy = min(max(gy + 1, 0), height - 1);
    tile[ty + 2][tx + 1] = d_input[hy * width + clamped_gx];
  }

  // Load corner halos
  if (tx == 0 && ty == 0) {
    tile[0][0] = d_input[max(gy - 1, 0) * width + max(gx - 1, 0)];
  }
  if (tx == blockDim.x - 1 && ty == 0) {
    tile[0][tx + 2] = d_input[max(gy - 1, 0) * width + min(gx + 1, width - 1)];
  }
  if (tx == 0 && ty == blockDim.y - 1) {
    tile[ty + 2][0] = d_input[min(gy + 1, height - 1) * width + max(gx - 1, 0)];
  }
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
    tile[ty + 2][tx + 2] = d_input[min(gy + 1, height - 1) * width + min(gx + 1, width - 1)];
  }

  __syncthreads();

  if (gx >= width || gy >= height) return;

  // 3x3 Gaussian Kernel approximation:
  // [ 1  2  1 ]
  // [ 2  4  2 ]  / 16
  // [ 1  2  1 ]
  int sum = 1 * tile[ty][tx]     + 2 * tile[ty][tx + 1]     + 1 * tile[ty][tx + 2] +
            2 * tile[ty + 1][tx] + 4 * tile[ty + 1][tx + 1] + 2 * tile[ty + 1][tx + 2] +
            1 * tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + 1 * tile[ty + 2][tx + 2];

  d_output[gy * width + gx] = static_cast<uint8_t>(sum >> 4);
}

__global__ void SobelEdgeSharedMemKernel(const uint8_t* __restrict__ d_input,
                                        uint8_t* __restrict__ d_output,
                                        int width, int height) {
  __shared__ uint8_t tile[SHARED_H][SHARED_W];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = blockIdx.x * blockDim.x + tx;
  int gy = blockIdx.y * blockDim.y + ty;

  int clamped_gx = min(max(gx, 0), width - 1);
  int clamped_gy = min(max(gy, 0), height - 1);
  tile[ty + 1][tx + 1] = d_input[clamped_gy * width + clamped_gx];

  if (tx == 0) {
    tile[ty + 1][0] = d_input[clamped_gy * width + max(gx - 1, 0)];
  }
  if (tx == blockDim.x - 1 || gx == width - 1) {
    tile[ty + 1][tx + 2] = d_input[clamped_gy * width + min(gx + 1, width - 1)];
  }
  if (ty == 0) {
    tile[0][tx + 1] = d_input[max(gy - 1, 0) * width + clamped_gx];
  }
  if (ty == blockDim.y - 1 || gy == height - 1) {
    tile[ty + 2][tx + 1] = d_input[min(gy + 1, height - 1) * width + clamped_gx];
  }
  if (tx == 0 && ty == 0) {
    tile[0][0] = d_input[max(gy - 1, 0) * width + max(gx - 1, 0)];
  }
  if (tx == blockDim.x - 1 && ty == 0) {
    tile[0][tx + 2] = d_input[max(gy - 1, 0) * width + min(gx + 1, width - 1)];
  }
  if (tx == 0 && ty == blockDim.y - 1) {
    tile[ty + 2][0] = d_input[min(gy + 1, height - 1) * width + max(gx - 1, 0)];
  }
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
    tile[ty + 2][tx + 2] = d_input[min(gy + 1, height - 1) * width + min(gx + 1, width - 1)];
  }

  __syncthreads();

  if (gx >= width || gy >= height) return;

  // Horizontal gradient Gx
  int gx_val = -1 * tile[ty][tx]     + 1 * tile[ty][tx + 2] +
               -2 * tile[ty + 1][tx] + 2 * tile[ty + 1][tx + 2] +
               -1 * tile[ty + 2][tx] + 1 * tile[ty + 2][tx + 2];

  // Vertical gradient Gy
  int gy_val = -1 * tile[ty][tx]     - 2 * tile[ty][tx + 1]     - 1 * tile[ty][tx + 2] +
                1 * tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + 1 * tile[ty + 2][tx + 2];

  float mag = sqrtf(static_cast<float>(gx_val * gx_val + gy_val * gy_val));
  d_output[gy * width + gx] = static_cast<uint8_t>(fminf(mag, 255.0f));
}

__global__ void ThresholdBinarizationKernel(uint8_t* __restrict__ d_data,
                                           int total_pixels,
                                           uint8_t threshold_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;

  d_data[idx] = (d_data[idx] >= threshold_val) ? 255 : 0;
}
