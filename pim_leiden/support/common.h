#ifndef _LEIDEN_COMMON_H_
#define _LEIDEN_COMMON_H_

/*
 * Leiden 公共参数（Host 与 DPU 共享）
 * - Host 将该结构体填好后拷贝到每个 DPU 的 MRAM 堆起始处
 * - DPU 端根据 phase 字段选择执行哪个阶段
 *   PHASE_LOCAL_MOVE    ：局部移动（Local Moving），只在本分片负责的社区范围内处理
 *   PHASE_REFINEMENT    ：精炼（Refinement），在社区诱导子图上做连通性细化
 *   PHASE_AGGREGATION   ：仅占位（聚合在 CPU 上实现，DPU 不处理）
 */

#include <stdint.h>

#define ROUND_UP_TO_MULTIPLE_OF_2(x)    ((((x) + 1)/2)*2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)
#define ROUND_UP_TO_MULTIPLE_OF_64(x)   ((((x) + 63)/64)*64)

#define setBit(val, idx) (val) |= (1 << (idx))
#define isSet(val, idx)  ((val) & (1 << (idx)))

enum LeidenPhase {
    PHASE_LOCAL_MOVE   = 0,
    PHASE_REFINEMENT   = 1,
    PHASE_AGGREGATION  = 2
};

/* 统一结构体布局：禁用隐式填充（对齐按 1 字节打包），确保 Host 与 DPU 一致 */
#if defined(__GNUC__)
#pragma pack(push, 1)
#endif

/*
 * 说明：
 *   - 所有 *_m 字段均为 MRAM 堆上的偏移（从 DPU_MRAM_HEAP_POINTER 起算的字节偏移）
 *   - localNodeList_m        ：本 DPU 分片管理的“局部节点列表”（元素为全局节点 ID，升序或任意皆可）
 *   - localNodePtr_m         ：本 DPU 局部 CSR 的行指针，长度 = numLocalNodes + 1
 *   - localNeighborIdxs_m    ：本 DPU 局部 CSR 的邻接拼接，元素为全局节点 ID
 *   - nodeCommunity_m        ：全局节点 -> 社区 的映射，长度 = numGlobalNodes
 *   - communityTotalDegree_m ：每个社区的总度 Σ_tot[c]，长度 = numCommunities（无权图即节点度之和）
 *   - activeNodes_m          ：本分片局部节点的“活跃位图”（可选），长度 = ceil(numLocalNodes/64) 个 uint64_t
 *   - perCommunityComponentCount_m：每个社区的连通分量个数（Refinement 阶段由 DPU 统计回写）
 *   - perNodeComponentIndex_m：每个局部节点的“分量代表标签”（可选），本示例未在 Host 使用
 *
 *   - managedCommStart / managedCommCount ：本分片负责的社区区间 [start, start+count)
 *   - movedFlag_m            ：阶段内是否发生了任何更新（0/1）
 */

struct LeidenParams {
    /* 规模信息 */
    uint32_t numGlobalNodes;              /* 全局节点数 */
    uint32_t numLocalNodes;               /* 本分片局部节点数 */
    uint32_t numCommunities;              /* 全局社区数 */

    /* 阶段控制 */
    uint32_t phase;                       /* 当前阶段（见 LeidenPhase） */
    uint32_t iteration;                   /* 迭代编号（调试用） */
    uint32_t maxIterations;               /* 阶段内迭代上限（Refinement 用） */

    /* MRAM 指针（偏移） */
    uint32_t movedFlag_m;                 /* 4B：是否有更新 */
    uint32_t localNodeList_m;             /* uint32_t[numLocalNodes] */
    uint32_t localNodePtr_m;              /* uint32_t[numLocalNodes+1] */
    uint32_t localNeighborIdxs_m;         /* uint32_t[sum(deg(local))] */
    uint32_t nodeCommunity_m;             /* uint32_t[numGlobalNodes] */
    uint32_t communityTotalDegree_m;      /* uint64_t[numCommunities] */
    uint32_t activeNodes_m;               /* uint64_t[ceil(numLocalNodes/64)] （可选） */

    /* Refinement 输出/工作区 */
    uint32_t perCommunityComponentCount_m;/* uint32_t[numCommunities] */
    uint32_t perNodeComponentIndex_m;     /* uint32_t[numLocalNodes] （可选） */

    /* 本分片负责的社区区间 */
    uint32_t managedCommStart;
    uint32_t managedCommCount;

    /* 模块度或增益计算相关（无权图时 twoM = 2*|E|） */
    uint64_t twoM;
};
#if defined(__GNUC__)
#pragma pack(pop)
#endif

#endif