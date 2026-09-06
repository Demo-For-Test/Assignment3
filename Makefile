NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 --std=c++17 -I./include -Xcompiler -Wall
CXX_FLAGS = -O3 --std=c++17 -I./include -Wall

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

TARGET = $(BIN_DIR)/batch_image_pipeline

OBJS = $(OBJ_DIR)/cuda_kernels.o \
       $(OBJ_DIR)/image_io.o \
       $(OBJ_DIR)/pipeline.o \
       $(OBJ_DIR)/main.o

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) data/output logs

$(OBJ_DIR)/cuda_kernels.o: $(SRC_DIR)/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/image_io.o: $(SRC_DIR)/image_io.cpp include/image_io.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/pipeline.o: $(SRC_DIR)/pipeline.cu include/pipeline.h include/cuda_kernels.h include/image_io.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp include/pipeline.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@

run: $(TARGET)
	./$(TARGET) --input-dir data/input --output-dir data/output --streams 4 --threshold 75

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
