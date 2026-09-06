// Copyright 2026 Enterprise CUDA Capstone Project Authors. All Rights Reserved.

#include "image_io.h"
#include <fstream>
#include <iostream>
#include <sstream>

static void SkipComments(std::ifstream& infile) {
  char ch;
  while (infile >> std::ws && (ch = infile.peek()) == '#') {
    std::string comment;
    std::getline(infile, comment);
  }
}

bool ReadPPM_P6(const std::string& filepath, ImageRGB* image) {
  if (!image) return false;
  std::ifstream infile(filepath, std::ios::binary);
  if (!infile.is_open()) return false;

  std::string magic;
  infile >> magic;
  if (magic != "P6") return false;

  SkipComments(infile);
  int w, h, max_val;
  infile >> w >> h;
  SkipComments(infile);
  infile >> max_val;
  infile.get(); // consume single whitespace after max_val

  image->width = w;
  image->height = h;
  image->data.resize(w * h * 3);

  infile.read(reinterpret_cast<char*>(image->data.data()), image->data.size());
  return infile.good();
}

bool WritePGM_P5(const std::string& filepath, const ImageGray& image) {
  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) return false;

  outfile << "P5\n" << image.width << " " << image.height << "\n255\n";
  outfile.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
  return outfile.good();
}

bool WritePPM_P6(const std::string& filepath, const ImageRGB& image) {
  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) return false;

  outfile << "P6\n" << image.width << " " << image.height << "\n255\n";
  outfile.write(reinterpret_cast<const char*>(image.data.data()), image.data.size());
  return outfile.good();
}
