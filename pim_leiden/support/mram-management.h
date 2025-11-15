#ifndef _MRAM_MANAGEMENT_H_
#define _MRAM_MANAGEMENT_H_

#include "../support/common.h"
#include "../support/utils.h"
#include <time.h>
#include <dpu.h>

#define DPU_CAPACITY (64u << 20)

extern double g_cpu_to_dpu_time;
extern double g_dpu_to_cpu_time;

static inline double _now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct mram_heap_allocator_t {
    uint32_t totalAllocated;
};

static inline void init_allocator(struct mram_heap_allocator_t* allocator) {
    allocator->totalAllocated = 0;
}

static inline uint32_t mram_heap_alloc(struct mram_heap_allocator_t* allocator, uint32_t size) {
    allocator->totalAllocated = ROUND_UP_TO_MULTIPLE_OF_8(allocator->totalAllocated);
    uint32_t ret = allocator->totalAllocated;
    allocator->totalAllocated += ROUND_UP_TO_MULTIPLE_OF_8(size);
    if (allocator->totalAllocated > DPU_CAPACITY) {
        PRINT_ERROR("Total memory allocated %u exceeds DPU capacity %u!", allocator->totalAllocated, DPU_CAPACITY);
        exit(1);
    }
    return ret;
}

static inline void copyToDPU(struct dpu_set_t dpu, const uint8_t* hostPtr, uint32_t mramIdx, uint32_t size) {
    uint32_t sz = ROUND_UP_TO_MULTIPLE_OF_8(size);
    double t0 = _now_seconds();
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, hostPtr, sz));
    double t1 = _now_seconds();
    g_cpu_to_dpu_time += (t1 - t0);
}

static inline void copyFromDPU(struct dpu_set_t dpu, uint32_t mramIdx, uint8_t* hostPtr, uint32_t size) {
    uint32_t sz = ROUND_UP_TO_MULTIPLE_OF_8(size);
    double t0 = _now_seconds();
    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, hostPtr, sz));
    double t1 = _now_seconds();
    g_dpu_to_cpu_time += (t1 - t0);
}

/* 广播：按需做 8 字节对齐填充 */
static inline void broadcastToAllDPUs(struct dpu_set_t dpu_set,
                                      const uint8_t* hostPtr,
                                      uint32_t mramIdx,
                                      uint32_t size) {
    uint32_t sz = ROUND_UP_TO_MULTIPLE_OF_8(size);
    /* 若 size 已对齐，直接广播；否则构造临时缓冲补零 */
    if (sz == size) {
        double t0 = _now_seconds();
        DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, hostPtr, sz, DPU_XFER_DEFAULT));
        double t1 = _now_seconds();
        g_cpu_to_dpu_time += (t1 - t0);
    } else {
        uint8_t *tmp = (uint8_t*)calloc(sz, 1);
        memcpy(tmp, hostPtr, size);
        double t0 = _now_seconds();
        DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, tmp, sz, DPU_XFER_DEFAULT));
        double t1 = _now_seconds();
        g_cpu_to_dpu_time += (t1 - t0);
        free(tmp);
    }
}

#endif