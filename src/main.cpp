// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#include <iostream>
#include <iomanip>
#include <string>
#include "pipeline.h"

void PrintUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --input-dir <path>     Directory containing input PPM images (default: data/input)\n"
            << "  --output-dir <path>    Directory for processed edge maps (default: data/output)\n"
            << "  --streams <N>          Number of concurrent CUDA streams (default: 4)\n"
            << "  --block-x <N>          Block Dim X for 2D kernels (default: 16)\n"
            << "  --block-y <N>          Block Dim Y for 2D kernels (default: 16)\n"
            << "  --threshold <N>        Edge binarization threshold 0-255 (default: 75)\n"
            << "  --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
  PipelineConfig config;
  config.input_dir = "data/input";
  config.output_dir = "data/output";
  config.num_streams = 4;
  config.block_dim_x = 16;
  config.block_dim_y = 16;
  config.edge_threshold = 75;
  config.enable_pinned_memory = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input-dir" && i + 1 < argc) {
      config.input_dir = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--streams" && i + 1 < argc) {
      config.num_streams = std::stoi(argv[++i]);
    } else if (arg == "--block-x" && i + 1 < argc) {
      config.block_dim_x = std::stoi(argv[++i]);
    } else if (arg == "--block-y" && i + 1 < argc) {
      config.block_dim_y = std::stoi(argv[++i]);
    } else if (arg == "--threshold" && i + 1 < argc) {
      config.edge_threshold = static_cast<uint8_t>(std::stoi(argv[++i]));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  std::cout << "======================================================================\n";
  std::cout << "   Enterprise CUDA Batch Image Processing Pipeline (Capstone Project) \n";
  std::cout << "======================================================================\n";
  std::cout << "Input Directory  : " << config.input_dir << "\n";
  std::cout << "Output Directory : " << config.output_dir << "\n";
  std::cout << "CUDA Streams     : " << config.num_streams << "\n";
  std::cout << "Block Dimensions : (" << config.block_dim_x << ", " << config.block_dim_y << ")\n";
  std::cout << "Edge Threshold   : " << static_cast<int>(config.edge_threshold) << "\n\n";

  BatchImagePipeline pipeline(config);
  BenchmarkMetrics metrics = pipeline.Run();

  std::cout << "\n======================================================================\n";
  std::cout << "                     EXECUTION BENCHMARK RESULTS                      \n";
  std::cout << "======================================================================\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Total Images Processed : " << metrics.total_images_processed << "\n";
  std::cout << "Total Elapsed Time     : " << metrics.total_time_ms << " ms (" 
            << metrics.total_time_ms / 1000.0 << " s)\n";
  std::cout << "Average Latency / Image: " << metrics.avg_latency_per_image_ms << " ms\n";
  std::cout << "Image Throughput       : " << metrics.throughput_img_per_sec << " images/sec\n";
  std::cout << "Effective Bandwidth    : " << metrics.throughput_mbytes_per_sec << " MB/s\n";
  std::cout << "======================================================================\n";

  return 0;
}
