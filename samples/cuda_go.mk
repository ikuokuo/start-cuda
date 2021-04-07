MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MK_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

INCLUDE_DIR := $(MK_DIR)/../include/
COMMON_FILES := $(wildcard $(INCLUDE_DIR)/common/*.hpp)

EXTRA_NVCCFLAGS ?= -std=c++11

include $(MK_DIR)/cuda.mk

INCLUDES += -I$(INCLUDE_DIR)

TGT_DIR := $(BUILD_TYPE)/$(TARGET_OS)-$(TARGET_ARCH)
BIN_DIR ?= $(MK_DIR)/$(TGT_DIR)/bin
OBJ_DIR ?= $(MK_DIR)/$(TGT_DIR)/obj

# host compiler
HOST_CCFLAGS ?= -std=c++11
HOST_LDFLAGS ?= $(HOST_CCFLAGS)

define echo
	text="$1"; options="$2"; \
	[ -z "$2" ] && options="1;32"; \
	echo "\033[$${options}m$${text}\033[0m"
endef

define cu2obj
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(OBJ_DIR)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $1 -o $@ -c $<
endef

define cuobj2bin
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(BIN_DIR)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $1 -o $@ $+ $(LIBRARIES)
endef

define cc2obj
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(OBJ_DIR)
	$(EXEC) $(HOST_COMPILER) $(INCLUDES) $(HOST_CCFLAGS) $1 -o $@ -c $<
endef

define ccobj2bin
	@$(call echo,$@ < $<)
	$(EXEC) @mkdir -p $(BIN_DIR)
	$(EXEC) $(HOST_COMPILER) $(HOST_LDFLAGS) $1 -o $@ $+
endef

define run
	$(EXEC) ./$(BIN_DIR)/$(1)
endef

define clean
	-rm -rf $(BIN_DIR)/$(1) $(OBJ_DIR)/$(1).o
endef

$(OBJ_DIR)/%.o: %.cu $(COMMON_FILES)
	$(call cu2obj)

$(OBJ_DIR)/%.o: %.cc $(COMMON_FILES)
	$(call cc2obj)
