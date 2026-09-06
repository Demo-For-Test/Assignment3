// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <string>
#include <vector>
#include <chrono>
#include "image_io.h"

struct PipelineConfig {
  std::string input_dir;
  std::string output_dir;
  int num_streams;
  int block_dim_x;
  int block_dim_y;
  uint8_t edge_threshold;
  bool enable_pinned_memory;
};

struct BenchmarkMetrics {
  int total_images_processed;
  double total_time_ms;
  double throughput_img_per_sec;
  double throughput_mbytes_per_sec;
  double avg_latency_per_image_ms;
};

class BatchImagePipeline {
 public:
  explicit BatchImagePipeline(const PipelineConfig& config);
  ~BatchImagePipeline();

  // Disallow copy and assign
  BatchImagePipeline(const BatchImagePipeline&) = delete;
  BatchImagePipeline& operator=(const BatchImagePipeline&) = delete;

  BenchmarkMetrics Run();

 private:
  PipelineConfig config_;
  std::vector<std::string> image_paths_;

  void DiscoverInputImages();
  void ProcessImageAsync(const std::string& in_path, const std::string& out_path,
                         cudaStream_t stream, uint8_t* d_rgb, uint8_t* d_gray,
                         uint8_t* d_blur, uint8_t* d_edge);
};

#endif  // PIPELINE_H_
