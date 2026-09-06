// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#ifndef IMAGE_IO_H_
#define IMAGE_IO_H_

#include <string>
#include <vector>
#include <cstdint>

struct ImageRGB {
  int width;
  int height;
  std::vector<uint8_t> data; // Size: width * height * 3
};

struct ImageGray {
  int width;
  int height;
  std::vector<uint8_t> data; // Size: width * height
};

// Reads a binary Netpbm Portable Pixmap (P6) format image file.
bool ReadPPM_P6(const std::string& filepath, ImageRGB* image);

// Writes a binary Netpbm Portable Graymap (P5) format image file.
bool WritePGM_P5(const std::string& filepath, const ImageGray& image);

// Writes a binary Netpbm Portable Pixmap (P6) format image file.
bool WritePPM_P6(const std::string& filepath, const ImageRGB& image);

#endif  // IMAGE_IO_H_
