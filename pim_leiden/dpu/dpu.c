/*
 * dpu/task.c
 * DPU 端内核（Leiden）：支持 Local Moving 与 Refinement 两个阶段
 *
 * 实现要点：
 *  - Local Moving（局部移动）：
 *      对本分片的“活跃节点”（如有 active 位图）或全部局部节点，统计邻居社区桶，估算增益（简化公式），
 *      若增益为正则将该节点迁移至对应社区。若任一节点迁移，则将 movedFlag 置 1。
 *  - Refinement（精炼）：
 *      在“同社区诱导子图”上做最小标签传播（label propagation of min label），最终每个连通分量内所有点拥有相同的最小标签。
 *      完成后统计每个社区的“代表点”数（即标签等于自身全局 ID 的节点数），写回 perCommunityComponentCount。
 *
 * 依赖：
 *  - inc/leiden-common.h：参数结构体
 *  - dpu-utils.h         ：MRAM 读写 4B/8B 小工具
 *  - ../support/common.h ：ROUND_UP 宏与基础类型
 */

#include <stdio.h>
#include <stdint.h>

#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <mutex.h>
#include <perfcounter.h>
#include <defs.h>

#include "../support/dpu-utils.h"
#include "../support/common.h"

/* 全局栅栏：用于 tasklets 同步 */
BARRIER_INIT(g_barrier, NR_TASKLETS);

/* 简化增益：gain = w_vc * twoM - k_v * Σ_tot[c]
 * - w_vc：v 与候选社区 c 的连接次数（无权图）
 * - k_v ：v 的度
 * - Σ_tot[c]：社区 c 总度
 * - twoM：2|E|
 * 注：此公式非严格模块度 ΔQ，仅用于示例，建议后续替换为标准 ΔQ。
 */
static inline int64_t simplified_gain(uint32_t w_vc, uint32_t k_v, uint64_t sumTot_c, uint64_t twoM) {
    return (int64_t)w_vc * (int64_t)twoM - (int64_t)k_v * (int64_t)sumTot_c;
}

/* 从 MRAM 的“局部节点列表（全局 ID）”中二分查找目标全局 ID，返回局部下标；未找到返回 -1 */
static inline int32_t mram_find_local_index(uint32_t base_m, uint32_t n, uint32_t target, uint64_t *cache_w) {
    int32_t lo = 0, hi = (int32_t)n - 1;
    while (lo <= hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        uint32_t midVal = load4B(base_m, (uint32_t)mid, cache_w);
        if (midVal == target) return mid;
        if (midVal < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ------------------------- Local Moving 实现 ------------------------- */

/* 每个 tasklet 的邻居社区“桶”上限（简单实现，避免动态分配） */
#ifndef NEIGHBOR_COMM_BUCKETS
#define NEIGHBOR_COMM_BUCKETS 64
#endif

struct CommBucket {
    uint32_t comm;    /* 社区 ID */
    uint32_t weight;  /* 计数（无权图为出现次数） */
};

static inline void bucket_reset(struct CommBucket *b) {
    for (int i=0; i<NEIGHBOR_COMM_BUCKETS; ++i) {
        b[i].comm = 0xFFFFFFFFu;
        b[i].weight = 0;
    }
}

static inline void bucket_add(struct CommBucket *b, uint32_t comm) {
    for (int i=0; i<NEIGHBOR_COMM_BUCKETS; ++i) {
        if (b[i].comm == comm) { b[i].weight += 1; return; }
        if (b[i].comm == 0xFFFFFFFFu) { b[i].comm = comm; b[i].weight = 1; return; }
    }
}

/* 判断局部节点 i 是否活跃（active 位图可选） */
static inline int is_local_active(uint32_t i, uint32_t numLocalNodes, uint32_t active_m, uint64_t *cache_w) {
    if (active_m == 0) return 1; /* 若未下发 active 位图，则默认全部活跃 */
    uint32_t word = i >> 6;
    if (word >= ((numLocalNodes + 63) >> 6)) return 0;
    uint64_t w = load8B(active_m, word, cache_w);
    return (int)((w >> (i & 63)) & 1ULL);
}

/* ------------------------- Refinement 实现 ------------------------- */

#define INVALID_LABEL 0xFFFFFFFFu
MUTEX_INIT(g_refine_mutex);

/* 标签初始化：同本分片管理社区范围内的节点，标签 = 自身全局 ID；其余节点 = INVALID_LABEL */
static inline void refine_init_labels(struct LeidenParams *P, uint64_t *cache_w) {
    uint32_t N = P->numLocalNodes;
    uint32_t cBeg = P->managedCommStart;
    uint32_t cEnd = cBeg + P->managedCommCount;

    /* tasklet 均分本地节点 */
    uint32_t per = (N + NR_TASKLETS - 1) / NR_TASKLETS;
    uint32_t s = me() * per;
    uint32_t e = s + per; if (e > N) e = N;

    for (uint32_t i=s; i<e; ++i) {
        uint32_t g = load4B(P->localNodeList_m, i, cache_w);
        uint32_t c = load4B(P->nodeCommunity_m, g, cache_w);
        uint32_t lab = (c >= cBeg && c < cEnd) ? g : INVALID_LABEL;
        /* 若 Host 未分配 perNodeComponentIndex_m，可忽略写；本示例保留接口 */
        if (P->perNodeComponentIndex_m) {
            store4B(lab, P->perNodeComponentIndex_m, i, cache_w);
        }
    }
}

/* 单次松弛：同社区诱导子图内取“最小标签” */
static inline uint32_t refine_relax_once(struct LeidenParams *P, uint64_t *cache_w) {
    uint32_t N = P->numLocalNodes;
    uint32_t cBeg = P->managedCommStart;
    uint32_t cEnd = cBeg + P->managedCommCount;
    uint32_t per = (N + NR_TASKLETS - 1) / NR_TASKLETS;
    uint32_t s = me() * per;
    uint32_t e = s + per; if (e > N) e = N;

    uint32_t changed = 0;

    for (uint32_t i=s; i<e; ++i) {
        uint32_t g = load4B(P->localNodeList_m, i, cache_w);
        uint32_t c = load4B(P->nodeCommunity_m, g, cache_w);
        if (!(c >= cBeg && c < cEnd)) continue;

        uint32_t lab_i = (P->perNodeComponentIndex_m) ? load4B(P->perNodeComponentIndex_m, i, cache_w) : g;
        if (lab_i == INVALID_LABEL) continue;

        uint32_t minLab = lab_i;

        uint32_t p0 = load4B(P->localNodePtr_m, i, cache_w);
        uint32_t p1 = load4B(P->localNodePtr_m, i + 1, cache_w);
        for (uint32_t eidx = p0; eidx < p1; ++eidx) {
            uint32_t ngh = load4B(P->localNeighborIdxs_m, eidx, cache_w);
            uint32_t c2  = load4B(P->nodeCommunity_m, ngh, cache_w);
            if (c2 != c) continue;

            int32_t j = mram_find_local_index(P->localNodeList_m, P->numLocalNodes, ngh, cache_w);
            if (j < 0) continue;

            uint32_t lab_j = (P->perNodeComponentIndex_m) ? load4B(P->perNodeComponentIndex_m, (uint32_t)j, cache_w) : ngh;
            if (lab_j != INVALID_LABEL && lab_j < minLab) minLab = lab_j;
        }

        if (minLab < lab_i && P->perNodeComponentIndex_m) {
            store4B(minLab, P->perNodeComponentIndex_m, i, cache_w);
            changed = 1;
        }
    }

    return changed;
}

/* 统计每社区的连通分量个数（统计“代表点”：标签 == 自身全局 ID） */
static inline void refine_count_components(struct LeidenParams *P, uint64_t *cache_w) {
    uint32_t N = P->numLocalNodes;
    uint32_t cBeg = P->managedCommStart;
    uint32_t cEnd = cBeg + P->managedCommCount;

    barrier_wait(&g_barrier);
    if (me() == 0) {
        /* 仅清零本分片负责的社区区间 */
        for (uint32_t c=cBeg; c<cEnd; ++c) {
            store4B(0u, P->perCommunityComponentCount_m, c, cache_w);
        }
    }
    barrier_wait(&g_barrier);

    uint32_t per = (N + NR_TASKLETS - 1) / NR_TASKLETS;
    uint32_t s = me() * per;
    uint32_t e = s + per; if (e > N) e = N;

    mutex_id_t mid = MUTEX_GET(g_refine_mutex);

    for (uint32_t i=s; i<e; ++i) {
        uint32_t g = load4B(P->localNodeList_m, i, cache_w);
        uint32_t c = load4B(P->nodeCommunity_m, g, cache_w);
        if (!(c >= cBeg && c < cEnd)) continue;

        uint32_t lab = (P->perNodeComponentIndex_m) ? load4B(P->perNodeComponentIndex_m, i, cache_w) : g;
        if (lab == g) {
            mutex_lock(mid);
            uint32_t cnt = load4B(P->perCommunityComponentCount_m, c, cache_w);
            store4B(cnt + 1, P->perCommunityComponentCount_m, c, cache_w);
            mutex_unlock(mid);
        }
    }
    barrier_wait(&g_barrier);
}

/* ------------------------- 主入口 ------------------------- */
int main() {
    if (me() == 0) mem_reset();
    barrier_wait(&g_barrier);

    /* 读取参数（位于 MRAM 堆起始） */
    uint32_t params_m = (uint32_t)DPU_MRAM_HEAP_POINTER;
    if (me() == 0) printf("DPU %u: params_m = %u\n", me(), params_m);
    struct LeidenParams *P = (struct LeidenParams*)mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct LeidenParams)));
    mram_read((__mram_ptr void const*)params_m, P, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct LeidenParams)));
    if (me() == 0) printf("twoM=%llu numLocal=%u numComm=%u\n", (unsigned long long)P->twoM, P->numLocalNodes, P->numCommunities); 

    if (P->numLocalNodes == 0) {
        return 0;
    }
    /* 8 字节 WRAM 缓存，用于 MRAM 的 8B 对齐读写 */
    uint64_t *cache_w = (uint64_t*)mem_alloc(sizeof(uint64_t));

    if (P->phase == PHASE_LOCAL_MOVE) {
        /* ----------------- Local Moving ----------------- */
        uint32_t N = P->numLocalNodes;
        uint32_t per = (N + NR_TASKLETS - 1) / NR_TASKLETS;
        uint32_t s = me() * per;
        uint32_t e = s + per; if (e > N) e = N;

        struct CommBucket *buckets = (struct CommBucket*)mem_alloc(sizeof(struct CommBucket) * NEIGHBOR_COMM_BUCKETS);
        uint32_t movedLocal = 0;

        for (uint32_t i=s; i<e; ++i) {
            /* 非活跃节点直接跳过（若未下发 active 位图，则默认全部活跃） */
            if (!is_local_active(i, P->numLocalNodes, P->activeNodes_m, cache_w)) continue;

            uint32_t g = load4B(P->localNodeList_m, i, cache_w);
            uint32_t p0 = load4B(P->localNodePtr_m, i, cache_w);
            uint32_t p1 = load4B(P->localNodePtr_m, i+1, cache_w);
            uint32_t k_v = (p1 - p0);
            if (k_v == 0) continue;

            uint32_t c_old = load4B(P->nodeCommunity_m, g, cache_w);

            bucket_reset(buckets);
            /* 统计邻居的社区出现次数 */
            for (uint32_t eidx = p0; eidx < p1; ++eidx) {
                uint32_t ngh = load4B(P->localNeighborIdxs_m, eidx, cache_w);
                uint32_t cng = load4B(P->nodeCommunity_m, ngh, cache_w);
                bucket_add(buckets, cng);
            }

            /* 遍历候选社区，选择增益最大的 */
            int64_t bestGain = 0;
            uint32_t bestComm = c_old;
            for (int bi=0; bi<NEIGHBOR_COMM_BUCKETS; ++bi) {
                if (buckets[bi].comm == 0xFFFFFFFFu) break;
                uint32_t cid = buckets[bi].comm;
                uint64_t sumTot = load8B(P->communityTotalDegree_m, cid, cache_w);
                int64_t gain = simplified_gain(buckets[bi].weight, k_v, sumTot, P->twoM);
                if (gain > bestGain && cid != c_old) { bestGain = gain; bestComm = cid; }
            }

            if (bestComm != c_old && bestGain > 0) {
                store4B(bestComm, P->nodeCommunity_m, g, cache_w);
                movedLocal = 1;
            }
        }
        if(me() == 0) printf("DPU Local Moving done. movelocal = %d\n", movedLocal);
        barrier_wait(&g_barrier);
        if (movedLocal) {
            store4B(1u, P->movedFlag_m, 0, cache_w);
        }
        barrier_wait(&g_barrier);
        return 0;
    }

    if (P->phase == PHASE_REFINEMENT) {
        /* ----------------- Refinement ----------------- */
        refine_init_labels(P, cache_w);
        barrier_wait(&g_barrier);

        /* 迭代标签传播，直到收敛或达到上限 */
        uint32_t maxIters = (P->maxIterations ? P->maxIterations : 64u);
        for (uint32_t it=0; it<maxIters; ++it) {
            barrier_wait(&g_barrier);
            if (me() == 0) store4B(0u, P->movedFlag_m, 0, cache_w);
            barrier_wait(&g_barrier);

            uint32_t chg = refine_relax_once(P, cache_w);
            if (chg) store4B(1u, P->movedFlag_m, 0, cache_w);

            barrier_wait(&g_barrier);
            uint32_t any = 0;
            if (me() == 0) any = load4B(P->movedFlag_m, 0, cache_w);
            barrier_wait(&g_barrier);
            if (!any) break;
        }
        barrier_wait(&g_barrier);

        refine_count_components(P, cache_w);
        barrier_wait(&g_barrier);
        return 0;
    }

    /* 聚合相应在 Host 实现，这里直接返回 */
    return 0;
}