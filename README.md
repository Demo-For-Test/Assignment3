# Enterprise CUDA Batch Image Processing Pipeline (Capstone Project)

An enterprise-grade, high-performance CUDA image processing pipeline built to process large batches of images concurrently using asynchronous CUDA streams, 2D shared memory tiling, and halo exchange.

---

## 🏛️ Project Architecture & Pipeline Flow

The pipeline ingests raw 24-bit RGB images (Netpbm PPM format), executes a 4-stage GPU processing graph, and outputs 8-bit edge gradient maps:

```
[Host Input Image Batch: 120 PPM Files]
                    |
                    v
 [cudaMemcpyAsync (H2D) across 4 CUDA Streams]
                    |
                    v
    +-----------------------------------------------+
    | Kernel 1: Perceptual Luma RGB->Grayscale      | (ITU-R BT.601)
    +-----------------------------------------------+
                    |
                    v
    +-----------------------------------------------+
    | Kernel 2: 3x3 Gaussian Blur (Shared Memory)   | (18x18 Tile + 1-px Halo)
    +-----------------------------------------------+
                    |
                    v
    +-----------------------------------------------+
    | Kernel 3: Sobel Edge Gradient (Shared Memory) | (Horizontal & Vertical Gx/Gy)
    +-----------------------------------------------+
                    |
                    v
    +-----------------------------------------------+
    | Kernel 4: Dynamic Thresholding / Binarization | (Contrast Normalization)
    +-----------------------------------------------+
                    |
                    v
 [cudaMemcpyAsync (D2H) across 4 CUDA Streams]
                    |
                    v
[Host Output Edge Maps: data/output/proc_*.pgm]
```

---

## 🚀 Key GPU Features & Technical Highlights

1. **Multi-Stream Asynchronous Concurrency**:
   - Manages $N$ concurrent `cudaStream_t` execution queues.
   - Overlaps Host-to-Device (H2D) PCIe memory transfers, compute kernel launches, and Device-to-Host (D2H) transfers, maximizing GPU utilization.
2. **Shared Memory Tiling with Halo Exchange**:
   - Gaussian and Sobel filters load $18 \times 18$ tiles into high-speed on-chip `__shared__` memory for a $16 \times 16$ output thread block.
   - Eliminates redundant global DRAM accesses across adjacent stencil neighbors, improving effective memory bandwidth.
3. **Google C++ Style Guide Compliant**:
   - Clean modular layout (`include/`, `src/`), `CamelCase` types, `snake_case` functions, strict type safety, zero raw pointer memory leaks.
4. **Comprehensive CLI Argument Parsing**:
   - Customizable via flags: `--input-dir`, `--output-dir`, `--streams`, `--block-x`, `--block-y`, `--threshold`.

---

## 🛠️ Build & Execution Instructions

### Prerequisites
- NVIDIA CUDA Toolkit 11.0+ (`nvcc`)
- GCC/G++ 9.0+ with C++17 support
- Make utility

### Compiling
```bash
make clean
make all
```

### Running the Pipeline
Run with default parameters (120 images, 4 streams, 16x16 blocks):
```bash
make run
```

Or execute directly with custom arguments:
```bash
./bin/batch_image_pipeline --input-dir data/input --output-dir data/output --streams 4 --threshold 75
```

Or via shell script:
```bash
chmod +x run.sh
./run.sh
```

---

## 📊 Proof of Execution Artifacts

Evidence of code execution on **120 images** (30 MB of image data) is stored in the repository:
- `data/input/`: 120 synthetic benchmark PPM images (256x256 RGB).
- `data/output/`: 120 processed edge-detected PGM images.
- `proof_of_execution/execution_log.txt`: Complete execution telemetry log showing per-stream scheduling and benchmark results.
- `proof_of_execution/`: Paired sample before/after images (`before_img_*.ppm` and `after_img_*_sobel_edge.pgm`).

---

## 📂 Repository Layout

```
├── Makefile                          # Build rules for nvcc and g++
├── run.sh                            # One-click build and execution script
├── README.md                         # Project overview and instructions
├── CAPSTONE_SUBMISSION.md            # Ready-to-paste Coursera submission text & video script
├── include/
│   ├── cuda_kernels.h                # GPU kernel declarations
│   ├── image_io.h                    # Netpbm P5/P6 reader & writer declarations
│   └── pipeline.h                    # Batch pipeline engine interface
├── src/
│   ├── cuda_kernels.cu               # Implementations of RGB->Gray, Gaussian, Sobel, Threshold
│   ├── image_io.cpp                  # Binary PPM/PGM parsing implementation
│   ├── pipeline.cu                   # Multi-stream pipeline orchestration
│   └── main.cpp                      # CLI argument driver and telemetry reporter
├── data/
│   ├── input/                        # 120 input benchmark PPM images
│   └── output/                       # 120 processed edge map PGM images
├── scripts/
│   ├── generate_dataset.py           # Dataset generator script
│   └── run_pipeline.py               # Automated pipeline runner
└── proof_of_execution/               # Verification images and execution log
    ├── execution_log.txt             # Verified benchmark log
    ├── before_img_000.ppm
    ├── after_img_000_sobel_edge.pgm
    └── ...
```
