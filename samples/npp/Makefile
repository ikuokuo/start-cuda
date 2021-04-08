include ../conf.mk
include ../cuda_go.mk

MY_CCFLAGS := -std=c++14 -DTIME_COST -DMY_SAMPLES_DIR="$(shell pwd)/.."
MY_CCFLAGS += -I$(OPENCV_PATH)/include/opencv4 -I$(CUDA_PATH)/include
# MY_LDFLAGS := -L$(OPENCV_PATH)/lib -lopencv_world
MY_LDFLAGS := -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
MY_LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lnppc -lnppicc

HOST_CCFLAGS += $(MY_CCFLAGS)
HOST_LDFLAGS += $(MY_LDFLAGS) -Wl,-rpath $(OPENCV_PATH)/lib

NVCC_CCFLAGS += $(MY_CCFLAGS)
NVCC_HOST_LDFLAGS += $(MY_LDFLAGS) -rpath $(OPENCV_PATH)/lib

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/color_conversion

clean:
	$(call clean,color_conversion)

# color_conversion

$(BIN_DIR)/color_conversion: $(OBJ_DIR)/color_conversion.o
	$(call cuobj2bin)
