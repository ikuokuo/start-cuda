include ../cuda_go.mk

MY_CCFLAGS := -std=c++14 -DTIME_COST
HOST_CCFLAGS += $(MY_CCFLAGS)
NVCC_CCFLAGS += $(MY_CCFLAGS)

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/quick_start_cpu \
	$(BIN_DIR)/quick_start_gpu

clean:
	$(call clean,quick_start_cpu)
	$(call clean,quick_start_gpu)

# quick_start_cpu

$(BIN_DIR)/quick_start_cpu: $(OBJ_DIR)/quick_start_cpu.o
	$(call ccobj2bin)

# quick_start_gpu

$(BIN_DIR)/quick_start_gpu: $(OBJ_DIR)/quick_start_gpu.o
	$(call cuobj2bin)
