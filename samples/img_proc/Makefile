include ../cuda_go.mk

MY_CCFLAGS := -std=c++14 -DMY_SAMPLES_DIR="$(shell pwd)/.."
MY_LDFLAGS := -ljpeg

HOST_CCFLAGS += $(MY_CCFLAGS)
HOST_LDFLAGS += $(MY_LDFLAGS)

NVCC_CCFLAGS += $(MY_CCFLAGS)
NVCC_HOST_LDFLAGS += $(MY_LDFLAGS)

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/grayscale

clean:
	$(call clean,grayscale)

# grayscale

$(BIN_DIR)/grayscale: $(OBJ_DIR)/grayscale.o $(OBJ_DIR)/grayscale_gpu.o
	$(call cuobj2bin)
