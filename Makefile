# ======= SPARSE版本：无截断，稀疏社区映射 =======
DPU_DIR     := dpu
HOST_DIR    := host
SUPPORT_DIR := support
BUILDDIR ?= bin

NR_TASKLETS ?= 12
BATCH_SIZE ?= 1280000
MAX_NEIGHBOR_COMMS ?= 1200000

HOST_TARGET := $(BUILDDIR)/host
DPU_TARGET  := $(BUILDDIR)/dpu

SUPPORT_SOURCES := $(wildcard $(SUPPORT_DIR)/*.c)
# 只使用 host.c 和 host_helpers.c
HOST_SOURCES := $(HOST_DIR)/host.c $(HOST_DIR)/host_helpers.c $(SUPPORT_SOURCES)

CFLAGS_COMMON := -Wall -Wextra -O3 -g -fopenmp
HOST_INCS := -I/usr/include/dpu -Isupport
HOST_LIBS := -ldpu -fopenmp

DPU_CC := dpu-upmem-dpurte-clang
DPU_CFLAGS := -Wall -Wextra -O2 -g \
              -DNR_TASKLETS=$(NR_TASKLETS) \
              -DBATCH_SIZE=$(BATCH_SIZE) \
              -DMAX_NEIGHBOR_COMMS=$(MAX_NEIGHBOR_COMMS)

.PHONY: all clean help

all: $(HOST_TARGET) $(DPU_TARGET)

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(HOST_TARGET): $(BUILDDIR) $(HOST_SOURCES)
	$(CC) -o $@ $(HOST_SOURCES) $(CFLAGS_COMMON) $(HOST_INCS) $(HOST_LIBS) \
		-DNR_TASKLETS=$(NR_TASKLETS) \
		-DBATCH_SIZE=$(BATCH_SIZE) \
		-DMAX_NEIGHBOR_COMMS=$(MAX_NEIGHBOR_COMMS) \
		-DDPU_BINARY=\"$(abspath $(DPU_TARGET))\"

$(DPU_TARGET): $(BUILDDIR) $(DPU_DIR)/dpu.c
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $(DPU_DIR)/dpu.c
	@which dpu-strip >/dev/null 2>&1 && dpu-strip $@ -o $@ || echo "[WARN] dpu-strip not found; skipping strip"

clean:
	rm -rf $(BUILDDIR)

help:
	@echo "稀疏版本Makefile - 无全局映射截断"
	@echo ""
	@echo "可配置参数："
	@echo "  NR_TASKLETS=$(NR_TASKLETS)       - 每个DPU的tasklet数量"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)         - 增量处理批次大小（不再硬性截断！）"
	@echo "  MAX_NEIGHBOR_COMMS=$(MAX_NEIGHBOR_COMMS) - 邻居社区映射容量"
	@echo ""
	@echo "编译示例："
	@echo "  make"
	@echo "  make BATCH_SIZE=131072  # 更大批次"
	@echo ""
	@echo "运行示例："
	@echo "  DELTA_PERCENT=50 bin/host graph.txt 256 10 output.csv membership.txt"

