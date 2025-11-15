#include "aggregation.h"
#include "graph_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * 构建社区超图（无权版）：
 *   输入原图 g (graph_t)，节点社区 membership，社区总数 numCommunities
 *   输出 cg：节点=社区；边=存在跨社区连接的社区对（不含自环），无向以双向存储。
 * 说明：
 *   原图可能是无向但存储为单向（或双向）；此实现不依赖边的方向，只要出现一条 u->v 且 cu!=cv 即加入社区对。
 *   若同一社区对出现多条边，只保留一条（当前不记录权重）。若需要权重，可利用 PairEntry.cnt。
 */

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t cnt;
    uint8_t  used;
} PairEntry;

/* 64 位混合哈希再取 32 位，开放定址 */
static inline uint32_t pair_hash(uint32_t a, uint32_t b) {
    uint64_t h = ((uint64_t)a << 32) ^ (uint64_t)b;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (uint32_t)h;
}

static uint32_t next_pow2(uint32_t need) {
    if (need < 16) need = 16;
    uint32_t v = need - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

static void map_add(PairEntry *tab, uint32_t cap, uint32_t a, uint32_t b) {
    uint32_t h = pair_hash(a,b) & (cap - 1);
    for (;;) {
        PairEntry *slot = &tab[h];
        if (!slot->used) {
            slot->used = 1;
            slot->a = a; slot->b = b;
            slot->cnt = 1;
            return;
        } else if (slot->a == a && slot->b == b) {
            slot->cnt += 1;
            return;
        }
        h = (h + 1) & (cap - 1);
    }
}

void buildCommunityGraph(const graph_t *g,
                         const uint32_t *membership,
                         uint32_t numCommunities,
                         graph_t *cg)
{
    /* 释放旧内容 */
    if (cg && cg->row_ptr) {
        free_graph(cg);
        cg->new_to_old = NULL;
    }
    if (!cg) return;
    memset(cg, 0, sizeof(*cg));

    if (!g || !membership || numCommunities == 0) {
        return;
    }

    /* 预估社区对数量：粗略取原始边数的一半（只为容量预估，不准确也没关系） */
    uint32_t approx_pairs = (g->n_edges / 2) + 1;
    uint32_t cap = next_pow2(approx_pairs * 2); /* 负载率 < 0.5 */

    PairEntry *tab = (PairEntry*)calloc(cap, sizeof(PairEntry));
    if (!tab) return;

    /* 遍历每条边形成社区对 */
    for (uint32_t u = 0; u < g->n_nodes; ++u) {
        uint32_t cu = membership[u];
        uint32_t s = g->row_ptr[u];
        uint32_t e = g->row_ptr[u + 1];
        for (uint32_t p = s; p < e; ++p) {
            uint32_t v = g->col_idx[p];
            uint32_t cv = membership[v];
            if (cu == cv) continue;
            uint32_t a = (cu < cv) ? cu : cv;
            uint32_t b = (cu < cv) ? cv : cu;
            map_add(tab, cap, a, b);
        }
    }

    /* 统计社区度（无向双向） */
    uint32_t n = numCommunities;
    uint32_t *deg = (uint32_t*)calloc(n, sizeof(uint32_t));
    if (!deg) { free(tab); return; }

    for (uint32_t i = 0; i < cap; ++i) {
        if (!tab[i].used) continue;
        deg[tab[i].a] += 1;
        deg[tab[i].b] += 1;
    }

    /* 构造行指针 */
    cg->row_ptr = (uint32_t*)calloc(n + 1, sizeof(uint32_t));
    if (!cg->row_ptr) {
        free(deg); free(tab);
        return;
    }
    for (uint32_t i = 0; i < n; ++i)
        cg->row_ptr[i + 1] = cg->row_ptr[i] + deg[i];

    cg->n_nodes = n;
    cg->n_edges = cg->row_ptr[n]; /* 双向边数量 */
    cg->col_idx = (uint32_t*)malloc(cg->n_edges * sizeof(uint32_t));
    if (!cg->col_idx) {
        free_graph(cg);
        free(deg); free(tab);
        return;
    }

    /* 填充邻接 */
    uint32_t *cur = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!cur) {
        free_graph(cg);
        free(deg); free(tab);
        return;
    }
    memcpy(cur, cg->row_ptr, n * sizeof(uint32_t));

    for (uint32_t i = 0; i < cap; ++i) {
        if (!tab[i].used) continue;
        uint32_t a = tab[i].a, b = tab[i].b;
        cg->col_idx[cur[a]++] = b;
        cg->col_idx[cur[b]++] = a;
    }

    free(cur);
    free(deg);
    free(tab);

    /* 社区图的 new_to_old：身份映射（社区ID保持不变） */
    cg->new_to_old = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (cg->new_to_old) {
        for (uint32_t i = 0; i < n; ++i) cg->new_to_old[i] = i;
    }
}