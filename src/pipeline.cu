// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#include "pipeline.h"
#include "cuda_kernels.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

BatchImagePipeline::BatchImagePipeline(const PipelineConfig& config)
    : config_(config) {
  DiscoverInputImages();
}

BatchImagePipeline::~BatchImagePipeline() {}

void BatchImagePipeline::DiscoverInputImages() {
  image_paths_.clear();
  if (!fs::exists(config_.input_dir)) {
    std::cerr << "Error: Input directory does not exist: " << config_.input_dir << std::endl;
    return;
  }

  for (const auto& entry : fs::directory_iterator(config_.input_dir)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".ppm" || ext == ".pnm") {
        image_paths_.push_back(entry.path().string());
      }
    }
  }

  std::sort(image_paths_.begin(), image_paths_.end());
  std::cout << "[Pipeline] Discovered " << image_paths_.size() 
            << " images in " << config_.input_dir << std::endl;
}

BenchmarkMetrics BatchImagePipeline::Run() {
  BenchmarkMetrics metrics = {0, 0.0, 0.0, 0.0, 0.0};
  if (image_paths_.empty()) {
    std::cerr << "[Pipeline] No input images found to process." << std::endl;
    return metrics;
  }

  fs::create_directories(config_.output_dir);

  int n_streams = std::max(1, config_.num_streams);
  std::vector<cudaStream_t> streams(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  // Pre-read first image to determine typical dimensions
  ImageRGB sample_img;
  if (!ReadPPM_P6(image_paths_[0], &sample_img)) {
    std::cerr << "Failed to read sample image for buffer sizing." << std::endl;
    return metrics;
  }

  int w = sample_img.width;
  int h = sample_img.height;
  size_t rgb_bytes = w * h * 3;
  size_t gray_bytes = w * h;

  // Allocate per-stream GPU device buffers
  struct StreamBuffers {
    uint8_t* d_rgb;
    uint8_t* d_gray;
    uint8_t* d_blur;
    uint8_t* d_edge;
  };

  std::vector<StreamBuffers> s_bufs(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    cudaMalloc(&s_bufs[i].d_rgb, rgb_bytes);
    cudaMalloc(&s_bufs[i].d_gray, gray_bytes);
    cudaMalloc(&s_bufs[i].d_blur, gray_bytes);
    cudaMalloc(&s_bufs[i].d_edge, gray_bytes);
  }

  dim3 block_dim(config_.block_dim_x, config_.block_dim_y);
  dim3 grid_dim((w + block_dim.x - 1) / block_dim.x,
                (h + block_dim.y - 1) / block_dim.y);

  std::cout << "[Pipeline] Starting execution across " << n_streams 
            << " CUDA streams with grid (" << grid_dim.x << "," << grid_dim.y 
            << ") and block (" << block_dim.x << "," << block_dim.y << ")...\n";

  auto start_wall = std::chrono::high_resolution_clock::now();

  size_t total_bytes_processed = 0;

  for (size_t i = 0; i < image_paths_.size(); ++i) {
    int s_idx = i % n_streams;
    cudaStream_t stream = streams[s_idx];
    StreamBuffers& bufs = s_bufs[s_idx];

    ImageRGB in_img;
    if (!ReadPPM_P6(image_paths_[i], &in_img)) {
      continue;
    }

    // 1. Asynchronous H2D Copy
    cudaMemcpyAsync(bufs.d_rgb, in_img.data.data(), rgb_bytes, cudaMemcpyHostToDevice, stream);

    // 2. Kernel 1: RGB -> Grayscale
    RgbToGrayscaleKernel<<<grid_dim, block_dim, 0, stream>>>(bufs.d_rgb, bufs.d_gray, w, h);

    // 3. Kernel 2: Shared Memory Gaussian Smoothing
    GaussianBlurSharedMemKernel<<<grid_dim, block_dim, 0, stream>>>(bufs.d_gray, bufs.d_blur, w, h);

    // 4. Kernel 3: Shared Memory Sobel Edge Detection
    SobelEdgeSharedMemKernel<<<grid_dim, block_dim, 0, stream>>>(bufs.d_blur, bufs.d_edge, w, h);

    // 5. Kernel 4: Contrast Normalization / Thresholding
    int total_pixels = w * h;
    int threads_1d = 256;
    int blocks_1d = (total_pixels + threads_1d - 1) / threads_1d;
    ThresholdBinarizationKernel<<<blocks_1d, threads_1d, 0, stream>>>(bufs.d_edge, total_pixels, config_.edge_threshold);

    // 6. Asynchronous D2H Copy
    ImageGray out_img;
    out_img.width = w;
    out_img.height = h;
    out_img.data.resize(gray_bytes);

    cudaMemcpyAsync(out_img.data.data(), bufs.d_edge, gray_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Write output file
    fs::path in_p(image_paths_[i]);
    std::string out_filename = "proc_" + in_p.filename().string();
    fs::path out_p = fs::path(config_.output_dir) / out_filename;
    WritePGM_P5(out_p.string(), out_img);

    total_bytes_processed += rgb_bytes + gray_bytes;
    metrics.total_images_processed++;
  }

  cudaDeviceSynchronize();
  auto end_wall = std::chrono::high_resolution_clock::now();

  metrics.total_time_ms = std::chrono::duration<double, std::milli>(end_wall - start_wall).count();
  metrics.throughput_img_per_sec = (metrics.total_images_processed / (metrics.total_time_ms / 1000.0));
  metrics.throughput_mbytes_per_sec = (total_bytes_processed / (1024.0 * 1024.0)) / (metrics.total_time_ms / 1000.0);
  metrics.avg_latency_per_image_ms = metrics.total_time_ms / metrics.total_images_processed;

  // Cleanup
  for (int i = 0; i < n_streams; ++i) {
    cudaFree(s_bufs[i].d_rgb);
    cudaFree(s_bufs[i].d_gray);
    cudaFree(s_bufs[i].d_blur);
    cudaFree(s_bufs[i].d_edge);
    cudaStreamDestroy(streams[i]);
  }

  return metrics;
}
