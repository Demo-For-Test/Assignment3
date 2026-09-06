// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");

#ifndef CUDA_KERNELS_H_
#define CUDA_KERNELS_H_

#include <cuda_runtime.h>
#include <cstdint>

// Converts 24-bit RGB (3 bytes/pixel) to 8-bit Grayscale (1 byte/pixel) using
// ITU-R BT.601 perceptual luma coefficients: Y = 0.299*R + 0.587*G + 0.114*B.
__global__ void RgbToGrayscaleKernel(const uint8_t* __restrict__ d_rgb,
                                    uint8_t* __restrict__ d_gray,
                                    int width, int height);

// Applies a 3x3 2D separable Gaussian smoothing filter using GPU Shared Memory tiles
// with halo exchange to maximize memory bandwidth and reduce global DRAM accesses.
__global__ void GaussianBlurSharedMemKernel(const uint8_t* __restrict__ d_input,
                                           uint8_t* __restrict__ d_output,
                                           int width, int height);

// Computes horizontal and vertical spatial derivatives using the 3x3 Sobel operator
// in Shared Memory and calculates the edge gradient magnitude: G = sqrt(Gx^2 + Gy^2).
__global__ void SobelEdgeSharedMemKernel(const uint8_t* __restrict__ d_input,
                                        uint8_t* __restrict__ d_output,
                                        int width, int height);

// Performs contrast normalization and dynamic thresholding on the edge map.
__global__ void ThresholdBinarizationKernel(uint8_t* __restrict__ d_data,
                                           int total_pixels,
                                           uint8_t threshold_val);

#endif  // CUDA_KERNELS_H_
