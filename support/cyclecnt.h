#include <perfcounter.h>

typedef struct perfcounter_cycles {
    perfcounter_t start;
} perfcounter_cycles;

static inline void timer_start(perfcounter_cycles *cycles) {
    cycles->start = perfcounter_get();
}

static inline uint64_t timer_stop(perfcounter_cycles *cycles) {
    perfcounter_t end = perfcounter_get();
    // 直接使用无符号差值，天然处理 32-bit 回绕
    return (uint32_t)(end - cycles->start);
}