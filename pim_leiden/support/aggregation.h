#ifndef _AGGREGATION_H_
#define _AGGREGATION_H_

#include <stdint.h>
#include "graph_loader.h"

/* 
 * 在 CPU 端构建“社区超图”：
 *   - 输入：原图 g，当前 membership（节点→社区），社区总数 numCommunities
 *   - 输出：社区图 cg（节点 = 社区，边 = 跨社区边的计数；不含自环）
 * 备注：
 *   - 无权场景下，边权为出现次数；若是加权图，可累加权重。
 */
void buildCommunityGraph(const graph_t *g,
                         const uint32_t *membership,
                         uint32_t numCommunities,
                         graph_t *cg);

#endif