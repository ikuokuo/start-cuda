include ../cuda_go.mk

MY_CCFLAGS := -std=c++14 -DTIME_COST
HOST_CCFLAGS += $(MY_CCFLAGS)
NVCC_CCFLAGS += $(MY_CCFLAGS)

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/matmul_cpu \
	$(BIN_DIR)/matmul_gpu \
	$(BIN_DIR)/matmul_gpu_shared

clean:
	$(call clean,matmul_cpu)
	$(call clean,matmul_gpu)
	$(call clean,matmul_gpu_shared)

# matmul_cpu

$(BIN_DIR)/matmul_cpu: $(OBJ_DIR)/matmul_cpu.o
	$(call ccobj2bin)

# matmul_gpu

$(BIN_DIR)/matmul_gpu: $(OBJ_DIR)/matmul_gpu.o
	$(call cuobj2bin)

# matmul_gpu_shared

$(BIN_DIR)/matmul_gpu_shared: $(OBJ_DIR)/matmul_gpu_shared.o
	$(call cuobj2bin)
