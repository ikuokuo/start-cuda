EXTRA_NVCCFLAGS := -std=c++11
EXTRA_LDFLAGS := -ljpeg

HOST_CCFLAGS := -std=c++11 -DMY_SAMPLES_DIR="$(shell pwd)/.."
HOST_LDFLAGS := $(HOST_CCFLAGS)

include ../cuda_go.mk

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/grayscale

clean:
	$(call clean,grayscale)

# grayscale

$(BIN_DIR)/grayscale: $(OBJ_DIR)/grayscale.o $(OBJ_DIR)/grayscale_gpu.o
	$(call cuobj2bin)
