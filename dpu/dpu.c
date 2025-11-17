/*
 * DPU kernel - Sparse community mapping版本
 * 不存储全局社区映射，只存储当前批次需要的邻居社区信息
 */
#include <stdint.h>
#include <stdbool.h>

#include <defs.h>
#include <mram.h>
#include <attributes.h>
#include <barrier.h>
#include <alloc.h>
#include "../support/cyclecnt.h"

#ifndef MAX_NODES_PER_DPU
#define MAX_NODES_PER_DPU 600000
#endif
#ifndef MAX_EDGES_PER_DPU
#define MAX_EDGES_PER_DPU 1200000
#endif
// 移除 MAX_GLOBAL_NODES 限制！
#ifndef MAX_COMM_CANDIDATES
#define MAX_COMM_CANDIDATES 64
#endif
// 增量处理：不再硬性限制，改为分批
#ifndef BATCH_SIZE
#define BATCH_SIZE 65536  // 每批处理的顶点数
#endif
// 邻居社区映射：只存储需要的部分
#ifndef MAX_NEIGHBOR_COMMS
#define MAX_NEIGHBOR_COMMS 1200000  // 与边数相当
#endif

#define INVALID_COMM 0xffffffffu
#define GAIN_EPSILON_SCALED 1

BARRIER_INIT(my_barrier, NR_TASKLETS);

// ========== Host控制变量 ==========
__host uint32_t delta_mode_active;
__host uint32_t base_id;
__host uint32_t node_count;
__host uint32_t changed_count;
__host uint32_t phase;
__host uint32_t two_m;
__host uint32_t batch_start_idx;  // 当前批次起始索引
__host uint32_t batch_count;      // 当前批次大小
__host uint32_t total_batches;    // 总批次数

// ========== 本地节点数据 (MRAM) ==========
__mram_noinit uint32_t neighbors[MAX_EDGES_PER_DPU];
__mram_noinit uint32_t node_row_ptr[MAX_NODES_PER_DPU + 1];
__mram_noinit uint32_t node_comm_local[MAX_NODES_PER_DPU];
__mram_noinit uint32_t node_comm_prev[MAX_NODES_PER_DPU];
__mram_noinit uint32_t node_degree_local[MAX_NODES_PER_DPU];

// ========== 稀疏社区映射 (MRAM) ==========
// 键值对：neighbor_id -> comm_id
__mram_noinit uint32_t neighbor_ids[MAX_NEIGHBOR_COMMS];
__mram_noinit uint32_t neighbor_comms[MAX_NEIGHBOR_COMMS];
__host uint32_t neighbor_map_size;

// 社区度数映射
__mram_noinit uint32_t comm_ids[MAX_NEIGHBOR_COMMS];
__mram_noinit uint32_t comm_degrees[MAX_NEIGHBOR_COMMS];
__host uint32_t comm_map_size;

// ========== 增量处理：当前批次顶点列表 ==========
__mram_noinit uint32_t delta_vertex_ids[BATCH_SIZE];

// ========== 变更输出 ==========
__mram_noinit uint32_t changed_global_ids[MAX_NODES_PER_DPU];
__mram_noinit uint32_t changed_new_comm[MAX_NODES_PER_DPU];
__mram_noinit uint32_t changed_degrees[MAX_NODES_PER_DPU];
__mram_noinit uint32_t changed_old_comm[MAX_NODES_PER_DPU];

// ========== 性能计数 ==========
__host uint64_t cycles_total;
__host uint64_t cycles_1;
__host uint64_t cycles_2;
__host uint64_t cycles_3;

// ========== WRAM工作区 ==========
typedef struct {
    uint32_t comm;
    uint32_t weight;
    uint32_t tot_deg;
} comm_entry_t;

__dma_aligned comm_entry_t comm_acc[NR_TASKLETS][MAX_COMM_CANDIDATES];
__dma_aligned uint32_t tasklet_counts[NR_TASKLETS];
__dma_aligned uint32_t prefix_offset[NR_TASKLETS];

#ifndef DMA_CHUNK
#define DMA_CHUNK 256
#endif

static uint32_t *g_nbr_bufs[NR_TASKLETS];

static inline uint32_t *tasklet_get_nbr_buf(void) {
    unsigned int tid = me();
    if (!g_nbr_bufs[tid]) {
        uint8_t *raw = (uint8_t *)mem_alloc(DMA_CHUNK * sizeof(uint32_t) + 8);
        uintptr_t aligned = ((uintptr_t)raw + 7u) & ~7u;
        g_nbr_bufs[tid] = (uint32_t *)aligned;
    }
    return g_nbr_bufs[tid];
}

// ========== 稀疏查找函数：二分查找 ==========
static inline uint32_t lookup_neighbor_comm(uint32_t node_id) {
    // 先检查是否是本地节点
    if (node_id >= base_id && node_id < base_id + node_count) {
        return node_comm_local[node_id - base_id];
    }
    
    // 在neighbor映射中二分查找
    uint32_t left = 0;
    uint32_t right = neighbor_map_size;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        uint32_t mid_id = neighbor_ids[mid];
        if (mid_id == node_id) {
            return neighbor_comms[mid];
        } else if (mid_id < node_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return node_id;  // 未找到，返回自身作为社区ID
}

static inline uint32_t lookup_comm_degree(uint32_t comm_id) {
    // 在comm度数映射中二分查找
    uint32_t left = 0;
    uint32_t right = comm_map_size;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        uint32_t mid_comm = comm_ids[mid];
        if (mid_comm == comm_id) {
            return comm_degrees[mid];
        } else if (mid_comm < comm_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return 0;
}

static inline void reset_acc(unsigned int tasklet_id) {
    for (uint32_t i = 0; i < MAX_COMM_CANDIDATES; ++i) {
        comm_acc[tasklet_id][i].comm = INVALID_COMM;
        comm_acc[tasklet_id][i].weight = 0;
        comm_acc[tasklet_id][i].tot_deg = 0;
    }
}

static inline void accumulate_comm(unsigned int tasklet_id, uint32_t comm_id) {
    int32_t slot = -1;
    int32_t empty_slot = -1;
    for (uint32_t i = 0; i < MAX_COMM_CANDIDATES; ++i) {
        uint32_t tracked = comm_acc[tasklet_id][i].comm;
        if (tracked == comm_id) {
            slot = (int32_t)i;
            break;
        }
        if (tracked == INVALID_COMM && empty_slot == -1) {
            empty_slot = (int32_t)i;
        }
    }

    if (slot != -1) {
        comm_acc[tasklet_id][slot].weight += 1;
        comm_acc[tasklet_id][slot].tot_deg = lookup_comm_degree(comm_id);
        return;
    }

    int32_t target = (empty_slot != -1) ? empty_slot : 0;
    if (empty_slot == -1) {
        uint32_t weakest_idx = 0;
        uint32_t weakest_weight = comm_acc[tasklet_id][0].weight;
        for (uint32_t i = 1; i < MAX_COMM_CANDIDATES; ++i) {
            if (comm_acc[tasklet_id][i].weight < weakest_weight) {
                weakest_weight = comm_acc[tasklet_id][i].weight;
                weakest_idx = i;
            }
        }
        target = (int32_t)weakest_idx;
    }
    comm_acc[tasklet_id][target].comm = comm_id;
    comm_acc[tasklet_id][target].weight = 1;
    comm_acc[tasklet_id][target].tot_deg = lookup_comm_degree(comm_id);
}

static inline uint32_t process_vertex(unsigned int tasklet_id,
                                      uint32_t local_index) {
    reset_acc(tasklet_id);
    uint32_t degree = node_degree_local[local_index];
    if (degree == 0) {
        node_comm_prev[local_index] = INVALID_COMM;
        return 0;
    }
    uint32_t current_comm = node_comm_local[local_index];
    node_comm_prev[local_index] = INVALID_COMM;

    uint32_t nbr_start = node_row_ptr[local_index];
    uint32_t nbr_end = node_row_ptr[local_index + 1];
    uint32_t *nbr_buf = tasklet_get_nbr_buf();
    
    for (uint32_t ei = nbr_start; ei < nbr_end; ei += DMA_CHUNK) {
        uint32_t chunk = (nbr_end - ei < DMA_CHUNK) ? (nbr_end - ei) : DMA_CHUNK;
        mram_read((__mram_ptr void const *)&neighbors[ei], nbr_buf, chunk * sizeof(uint32_t));
        for (uint32_t i = 0; i < chunk; ++i) {
            uint32_t neighbor = nbr_buf[i];
            uint32_t comm_id = lookup_neighbor_comm(neighbor);
            accumulate_comm(tasklet_id, comm_id);
        }
    }

    uint32_t current_weight = 0;
    for (uint32_t i = 0; i < MAX_COMM_CANDIDATES; ++i) {
        comm_entry_t entry = comm_acc[tasklet_id][i];
        if (entry.comm == current_comm) {
            current_weight = entry.weight;
            break;
        }
    }

    uint32_t current_tot = lookup_comm_degree(current_comm);
    uint32_t current_tot_after = (current_tot > degree) ? (current_tot - degree) : 0;
    
    int64_t base_loss_scaled = (int64_t)current_weight * (int64_t)two_m 
                              - (int64_t)degree * (int64_t)current_tot_after;

    int64_t best_delta_scaled = 0;
    uint32_t best_comm = current_comm;
    uint32_t best_weight = current_weight;

    for (uint32_t i = 0; i < MAX_COMM_CANDIDATES; ++i) {
        comm_entry_t entry = comm_acc[tasklet_id][i];
        if (entry.comm == INVALID_COMM || entry.weight == 0) {
            continue;
        }
        uint32_t candidate_comm = entry.comm;
        uint32_t candidate_tot = entry.tot_deg;
        if (candidate_comm == current_comm) {
            candidate_tot = (candidate_tot > degree) ? (candidate_tot - degree) : 0;
        }
        
        int64_t gain_scaled = (int64_t)entry.weight * (int64_t)two_m 
                             - (int64_t)degree * (int64_t)candidate_tot;
        int64_t delta_scaled = gain_scaled - base_loss_scaled;

        bool better = false;
        if (delta_scaled > best_delta_scaled + GAIN_EPSILON_SCALED) {
            better = true;
        } else if ((delta_scaled > best_delta_scaled - GAIN_EPSILON_SCALED) && (entry.weight > best_weight)) {
            better = true;
        } else if ((delta_scaled > best_delta_scaled - GAIN_EPSILON_SCALED) &&
                   (entry.weight == best_weight) &&
                   (candidate_comm < best_comm)) {
            better = true;
        }
        if (better) {
            best_delta_scaled = delta_scaled;
            best_comm = candidate_comm;
            best_weight = entry.weight;
        }
    }

    bool move = false;
    if (phase == 0) {
        move = (best_comm != current_comm) && (best_delta_scaled > GAIN_EPSILON_SCALED);
    } else {
        move = (best_comm != current_comm) && (current_weight == 0) && (best_weight > 0);
    }
    
    if (!move) {
        return 0;
    }

    node_comm_prev[local_index] = current_comm;
    node_comm_local[local_index] = best_comm;
    return 1;
}

int main() {
    unsigned int tasklet_id = me();
    perfcounter_config(COUNT_CYCLES, false);
    perfcounter_cycles cycles;
    perfcounter_cycles cycles_all;
    
    if (tasklet_id == 0) {
        timer_start(&cycles_all);
        timer_start(&cycles);
    }

    // 初始化缓冲区
    (void)tasklet_get_nbr_buf();

    if (tasklet_id == 0) {
        cycles_1 = timer_stop(&cycles);
        timer_start(&cycles);
    }

    barrier_wait(&my_barrier);

    uint32_t local_moves = 0;
    uint32_t local_batch_count = batch_count;  // 不再截断！
    
    if (local_batch_count > 0) {
        // 增量模式：只处理delta列表中的顶点
        for (uint32_t idx = tasklet_id; idx < local_batch_count; idx += NR_TASKLETS) {
            uint32_t global_node = delta_vertex_ids[idx];
            if (global_node < base_id || global_node >= base_id + node_count) {
                continue;
            }
            uint32_t fi = global_node - base_id;
            local_moves += process_vertex(tasklet_id, fi);
        }
    } else {
        // 全量模式：处理所有本地节点
        for (uint32_t fi = tasklet_id; fi < node_count; fi += NR_TASKLETS) {
            local_moves += process_vertex(tasklet_id, fi);
        }
    }
    
    tasklet_counts[tasklet_id] = local_moves;
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        cycles_2 = timer_stop(&cycles);
        timer_start(&cycles);
        uint32_t total = 0;
        for (uint32_t t = 0; t < NR_TASKLETS; ++t) {
            prefix_offset[t] = total;
            total += tasklet_counts[t];
        }
        changed_count = total;
    }

    barrier_wait(&my_barrier);

    // 收集变更
    uint32_t write_pos = prefix_offset[tasklet_id];
    for (uint32_t fi = tasklet_id; fi < node_count; fi += NR_TASKLETS) {
        uint32_t old_comm = node_comm_prev[fi];
        if (old_comm == INVALID_COMM) {
            continue;
        }
        uint32_t global_node = base_id + fi;
        uint32_t new_comm = node_comm_local[fi];
        uint32_t degree_val = node_degree_local[fi];
        
        changed_global_ids[write_pos] = global_node;
        changed_old_comm[write_pos] = old_comm;
        changed_new_comm[write_pos] = new_comm;
        changed_degrees[write_pos] = degree_val;
        write_pos++;
        node_comm_prev[fi] = new_comm;
    }

    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        cycles_3 = timer_stop(&cycles);
        cycles_total = timer_stop(&cycles_all);
    }
    
    return 0;
}

