#ifndef _GRAPH_LOADER_H_
#define _GRAPH_LOADER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    uint32_t *row_ptr;     // size = n_nodes + 1
    uint32_t *col_idx;     // size = n_edges
    uint32_t *new_to_old;  // size = n_nodes (原始ID映射)
    uint32_t  n_nodes;
    uint32_t  n_edges;
} graph_t;

/* 简单释放函数 */
static inline void free_graph(graph_t *g) {
    if(!g) return;
    free(g->row_ptr);
    free(g->col_idx);
    memset(g, 0, sizeof(*g));
}

/* 用于 qsort 的比较函数 */
static inline int compare_u32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t*)a;
    uint32_t ub = *(const uint32_t*)b;
    return (ua < ub) ? -1 : (ua > ub);
}

/* 读取图：你给出的实现（在 .c 中定义） */
bool load_graph(const char *filename, graph_t *graph);

#endif