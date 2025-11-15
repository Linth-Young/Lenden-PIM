# ======= simple UPMEM build =======
DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin

# 覆盖示例：make NR_TASKLETS=8
NR_TASKLETS ?= 12
NR_DPUS ?= 16

HOST_TARGET := $(BUILDDIR)/host
DPU_TARGET  := $(BUILDDIR)/dpu

HOST_SOURCES := $(wildcard $(HOST_DIR)/*.c)
DPU_SOURCES  := $(wildcard $(DPU_DIR)/*.c)

CFLAGS_COMMON := -Wall -Wextra -O3 -g -DNR_DPUS=$(NR_DPUS)
HOST_INCS := -I/usr/include/dpu -Isupport
HOST_LIBS := -ldpu

DPU_CC := dpu-upmem-dpurte-clang
DPU_CFLAGS := -Wall -Wextra -O2 -g -DNR_TASKLETS=$(NR_TASKLETS) 

.PHONY: all run_small run_mid clean
all: $(HOST_TARGET) $(DPU_TARGET)

# Separate target to run the delta experiments so `make` only builds by default.
run_mid: $(HOST_TARGET) $(DPU_TARGET)
	for p in 10 20; do \
		echo "\n=== Delta $${p}% ==="; \
		DELTA_PERCENT=$${p} $(HOST_TARGET) graph/germany_osm_edgelist.txt 1024 1 output/mid_$${p}.csv graph/germany_osm_part_leiden.txt > log/mid_$${p}.log; \
	done

run_small: $(HOST_TARGET) $(DPU_TARGET)
	for p in 10; do \
		echo "\n=== Delta $${p}% ==="; \
		DELTA_PERCENT=$${p} $(HOST_TARGET) graph/amazon_sym.txt graph/part_leiden.txt - 1 1 output/small_$${p}.csv > log/small_$${p}.log; \
	done

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

$(HOST_TARGET): $(BUILDDIR) $(HOST_SOURCES)
	$(CC) -o $@ $(HOST_SOURCES) $(CFLAGS_COMMON) $(HOST_INCS) $(HOST_LIBS) \
		-DNR_TASKLETS=$(NR_TASKLETS) \
		-DDPU_BINARY=\"$(abspath $(DPU_TARGET))\"

$(DPU_TARGET): $(BUILDDIR) $(DPU_SOURCES)
	$(DPU_CC) $(DPU_CFLAGS) -o $@ $(DPU_SOURCES)
	@which dpu-strip >/dev/null 2>&1 && dpu-strip $@ -o $@ || echo "[WARN] dpu-strip not found; skipping strip"

clean:
	rm -rf $(BUILDDIR)