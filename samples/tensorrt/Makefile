include ../conf.mk
include ../cuda_go.mk

MY_CCFLAGS := -std=c++14
MY_CCFLAGS += -I$(OPENCV_PATH)/include/opencv4 -I$(CUDA_PATH)/include -I$(TENSORRT_PATH)/include
MY_CCFLAGS += -Wno-deprecated-declarations
# MY_LDFLAGS := -L$(OPENCV_PATH)/lib -lopencv_world
MY_LDFLAGS := -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
MY_LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcudnn
MY_LDFLAGS += -L$(TENSORRT_PATH)/lib -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser
# MY_LDFLAGS += -lrt -ldl -lpthread

HOST_CCFLAGS += $(MY_CCFLAGS)
HOST_LDFLAGS += $(MY_LDFLAGS) -Wl,-rpath $(OPENCV_PATH)/lib

NVCC_CCFLAGS += $(MY_CCFLAGS)
NVCC_HOST_LDFLAGS += $(MY_LDFLAGS) -rpath $(OPENCV_PATH)/lib

.PHONY: all build clean

all: build

build: \
	$(BIN_DIR)/rvm_infer

clean:
	$(call clean,rvm_infer)

# rvm_infer

$(BIN_DIR)/rvm_infer: $(OBJ_DIR)/rvm_infer.o
	$(call ccobj2bin)
