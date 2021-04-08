MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

include $(MK_DIR)/cuda.mk

INCLUDE_DIR := $(MK_DIR)/../include/
INCLUDE_COMM_FILES := $(wildcard $(INCLUDE_DIR)/common/*.hpp)

COMM_CCFLAGS := -I$(INCLUDE_DIR)
HOST_CCFLAGS += $(COMM_CCFLAGS)
NVCC_CCFLAGS += $(COMM_CCFLAGS)

COMM_LDFLAGS :=
HOST_LDFLAGS += $(COMM_LDFLAGS)
NVCC_LDFLAGS += $(COMM_LDFLAGS)

TGT_DIR := $(BUILD_TYPE)/$(TARGET_OS)-$(TARGET_ARCH)
BIN_DIR ?= $(MK_DIR)/$(TGT_DIR)/bin
OBJ_DIR ?= $(MK_DIR)/$(TGT_DIR)/obj

define echo
	text="$1"; options="$2"; \
	[ -z "$2" ] && options="1;32"; \
	echo "\033[$${options}m$${text}\033[0m"
endef

define cu2obj
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(OBJ_DIR)
	$(EXEC) $(NVCC) $(NVCC_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@
endef

define cuobj2bin
	@$(call echo,$@ < $^)
	$(EXEC) @mkdir -p $(BIN_DIR)
	$(EXEC) $(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) $^ -o $@
endef

define cc2obj
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(OBJ_DIR)
	$(EXEC) $(HOST_COMPILER) $(HOST_CCFLAGS) -c $< -o $@
endef

define ccobj2bin
	@$(call echo,$@ < $^)
	$(EXEC) @mkdir -p $(BIN_DIR)
	$(EXEC) $(HOST_COMPILER) $^ $(HOST_LDFLAGS) -o $@
endef

define run
	$(EXEC) ./$(BIN_DIR)/$(1)
endef

define clean
	-rm -rf $(BIN_DIR)/$(1) $(OBJ_DIR)/$(1).o
endef

$(OBJ_DIR)/%.o: %.cu $(INCLUDE_COMM_FILES)
	$(call cu2obj)

$(OBJ_DIR)/%.o: %.cc $(INCLUDE_COMM_FILES)
	$(call cc2obj)
