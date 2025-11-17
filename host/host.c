/*
 * Host端 - 稀疏映射版本（多线程优化）
 * 配合 dpu_sparse.c 使用，支持无截断的分批处理
 */
#include <dpu.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "host_helpers.h"

#ifndef BATCH_SIZE
#define BATCH_SIZE 65536
#endif

#ifndef MAX_NEIGHBOR_COMMS
#define MAX_NEIGHBOR_COMMS 1200000
#endif

// 稀疏社区映射结构
typedef struct {
    uint32_t *neighbor_ids;      
    uint32_t *neighbor_comms;    
    uint32_t neighbor_count;     
    
    uint32_t *comm_ids;          
    uint32_t *comm_degrees;      
    uint32_t comm_count;         
} sparse_comm_map_t;

static void sparse_comm_map_init(sparse_comm_map_t *map) {
    map->neighbor_ids = NULL;
    map->neighbor_comms = NULL;
    map->neighbor_count = 0;
    map->comm_ids = NULL;
    map->comm_degrees = NULL;
    map->comm_count = 0;
}

static void sparse_comm_map_free(sparse_comm_map_t *map) {
    free(map->neighbor_ids);
    free(map->neighbor_comms);
    free(map->comm_ids);
    free(map->comm_degrees);
    sparse_comm_map_init(map);
}

static int compare_u32_pair(const void *a, const void *b) {
    uint32_t ua = ((const uint32_t *)a)[0];
    uint32_t ub = ((const uint32_t *)b)[0];
    return (ua < ub) ? -1 : (ua > ub);
}

// 为指定DPU构建稀疏社区映射（优化版本：使用哈希表而非全局数组）
static bool build_sparse_map_for_dpu(
    const uint32_t *vertices,
    uint32_t vertex_count,
    const graph_t *graph,
    const uint32_t *node_comm,
    const uint32_t *comm_tot_degree,
    uint32_t dpu_start_node,
    uint32_t dpu_node_count,
    sparse_comm_map_t *out_map)
{
    uint32_t *temp_neighbors = malloc(MAX_NEIGHBOR_COMMS * 2 * sizeof(uint32_t));
    uint32_t *temp_comms = malloc(MAX_NEIGHBOR_COMMS * 2 * sizeof(uint32_t));
    if (!temp_neighbors || !temp_comms) {
        free(temp_neighbors);
        free(temp_comms);
        return false;
    }

    uint32_t neighbor_idx = 0;
    uint32_t comm_idx = 0;
    
    // 优化：使用临时数组收集，然后排序去重，而不是大数组标记
    // 这样避免为大图分配巨大的seen数组

    // 遍历批次中的顶点，直接收集所有邻居和社区（允许重复）
    for (uint32_t i = 0; i < vertex_count; ++i) {
        uint32_t v = vertices[i];
        
        if (v < dpu_start_node || v >= dpu_start_node + dpu_node_count) {
            continue;
        }
        
        if (v >= graph->n_nodes) continue;
        
        uint32_t v_comm = node_comm[v];
        
        // 记录该顶点的社区（允许重复，稍后排序去重）
        if (v_comm < graph->n_nodes && comm_idx < MAX_NEIGHBOR_COMMS) {
            temp_comms[comm_idx * 2] = v_comm;
            temp_comms[comm_idx * 2 + 1] = comm_tot_degree[v_comm];
            comm_idx++;
        }
        
        // 遍历邻居
        uint32_t edge_start = graph->row_ptr[v];
        uint32_t edge_end = graph->row_ptr[v + 1];
        
        for (uint32_t ei = edge_start; ei < edge_end; ++ei) {
            if (neighbor_idx >= MAX_NEIGHBOR_COMMS) break;
            
            uint32_t neighbor = graph->col_idx[ei];
            
            // 如果邻居是外部节点，记录其社区（允许重复）
            if (neighbor < dpu_start_node || neighbor >= dpu_start_node + dpu_node_count) {
                if (neighbor < graph->n_nodes) {
                    uint32_t nbr_comm = node_comm[neighbor];
                    
                    temp_neighbors[neighbor_idx * 2] = neighbor;
                    temp_neighbors[neighbor_idx * 2 + 1] = nbr_comm;
                    neighbor_idx++;
                    
                    // 记录该社区的度数
                    if (nbr_comm < graph->n_nodes && comm_idx < MAX_NEIGHBOR_COMMS) {
                        temp_comms[comm_idx * 2] = nbr_comm;
                        temp_comms[comm_idx * 2 + 1] = comm_tot_degree[nbr_comm];
                        comm_idx++;
                    }
                }
            }
        }
    }

    // 排序并去重
    if (neighbor_idx > 0) {
        qsort(temp_neighbors, neighbor_idx, 2 * sizeof(uint32_t), compare_u32_pair);
        // 去重
        uint32_t unique_neighbors = 0;
        for (uint32_t i = 0; i < neighbor_idx; ++i) {
            if (i == 0 || temp_neighbors[i * 2] != temp_neighbors[(i - 1) * 2]) {
                temp_neighbors[unique_neighbors * 2] = temp_neighbors[i * 2];
                temp_neighbors[unique_neighbors * 2 + 1] = temp_neighbors[i * 2 + 1];
                unique_neighbors++;
            }
        }
        neighbor_idx = unique_neighbors;
    }
    
    if (comm_idx > 0) {
        qsort(temp_comms, comm_idx, 2 * sizeof(uint32_t), compare_u32_pair);
        // 去重
        uint32_t unique_comms = 0;
        for (uint32_t i = 0; i < comm_idx; ++i) {
            if (i == 0 || temp_comms[i * 2] != temp_comms[(i - 1) * 2]) {
                temp_comms[unique_comms * 2] = temp_comms[i * 2];
                temp_comms[unique_comms * 2 + 1] = temp_comms[i * 2 + 1];
                unique_comms++;
            }
        }
        comm_idx = unique_comms;
    }

    // 分配最终数组（8字节对齐以满足MRAM要求）
    out_map->neighbor_count = neighbor_idx;
    if (neighbor_idx > 0) {
        // 计算8字节对齐的大小（向上取整到8的倍数）
        size_t neighbor_bytes = neighbor_idx * sizeof(uint32_t);
        size_t aligned_bytes = ((neighbor_bytes + 7) / 8) * 8;
        
        out_map->neighbor_ids = malloc(aligned_bytes);
        out_map->neighbor_comms = malloc(aligned_bytes);
        if (!out_map->neighbor_ids || !out_map->neighbor_comms) {
            sparse_comm_map_free(out_map);
            free(temp_neighbors);
            free(temp_comms);
            return false;
        }
        
        // 初始化为0（包括padding区域）
        memset(out_map->neighbor_ids, 0, aligned_bytes);
        memset(out_map->neighbor_comms, 0, aligned_bytes);
        
        for (uint32_t i = 0; i < neighbor_idx; ++i) {
            out_map->neighbor_ids[i] = temp_neighbors[i * 2];
            out_map->neighbor_comms[i] = temp_neighbors[i * 2 + 1];
        }
    }
    
    out_map->comm_count = comm_idx;
    if (comm_idx > 0) {
        // 计算8字节对齐的大小
        size_t comm_bytes = comm_idx * sizeof(uint32_t);
        size_t aligned_comm_bytes = ((comm_bytes + 7) / 8) * 8;
        
        out_map->comm_ids = malloc(aligned_comm_bytes);
        out_map->comm_degrees = malloc(aligned_comm_bytes);
        if (!out_map->comm_ids || !out_map->comm_degrees) {
            sparse_comm_map_free(out_map);
            free(temp_neighbors);
            free(temp_comms);
            return false;
        }
        
        // 初始化为0（包括padding区域）
        memset(out_map->comm_ids, 0, aligned_comm_bytes);
        memset(out_map->comm_degrees, 0, aligned_comm_bytes);
        
        for (uint32_t i = 0; i < comm_idx; ++i) {
            out_map->comm_ids[i] = temp_comms[i * 2];
            out_map->comm_degrees[i] = temp_comms[i * 2 + 1];
        }
    }

    free(temp_neighbors);
    free(temp_comms);
    
    return true;
}

// 分批运行阶段（带完整性能统计）
static uint64_t run_phase_batched(
    struct dpu_set_t dpu_set,
    const dpu_partition_t *partitions,
    uint32_t num_dpus,
    const graph_t *graph,
    const uint32_t *node_comm,
    const uint32_t *comm_tot_degree,
    const uint32_t *vertices,
    uint32_t vertex_count,
    uint32_t phase,
    uint32_t two_m,
    u32_array_t *changed_nodes_out,
    u32_array_t *changed_comms_out,
    u32_array_t *changed_old_comms_out,
    u32_array_t *changed_degrees_out,
    double *dpu_time_accum)
{
    uint64_t total_moves = 0;
    uint32_t num_batches = (vertex_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    double batch_prep_time = 0.0;  // 批次准备时间（构建+传输映射）
    
    profile_printf("  分批处理: %u 顶点分成 %u 批 (batch_size=%u)\n",
                  vertex_count, num_batches, BATCH_SIZE);
    
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        uint32_t batch_start = batch * BATCH_SIZE;
        uint32_t batch_end = batch_start + BATCH_SIZE;
        if (batch_end > vertex_count) {
            batch_end = vertex_count;
        }
        uint32_t batch_count = batch_end - batch_start;
        
        profile_printf("    批次 %u/%u: %u 顶点 [%u:%u)\n",
                      batch + 1, num_batches, batch_count, batch_start, batch_end);
        
        // === 阶段1: 构建稀疏映射（多线程并行） ===
        double build_start = profile_time_now();
        
        // 为每个DPU准备稀疏映射
        sparse_comm_map_t *dpu_maps = malloc(num_dpus * sizeof(sparse_comm_map_t));
        if (!dpu_maps) {
            fprintf(stderr, "分配DPU映射数组失败\n");
            return UINT64_MAX;
        }
        
        // 初始化所有映射
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            sparse_comm_map_init(&dpu_maps[dpu_idx]);
        }
        
        // 并行构建映射
        int build_failed = 0;
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t dpu_idx = 0; dpu_idx < num_dpus; ++dpu_idx) {
            if (build_failed) continue;  // 如果已经有失败，跳过
            
            const dpu_partition_t *part = &partitions[dpu_idx];
            
            if (!build_sparse_map_for_dpu(
                    vertices + batch_start,
                    batch_count,
                    graph,
                    node_comm,
                    comm_tot_degree,
                    part->start_node,
                    part->local_nodes,
                    &dpu_maps[dpu_idx])) {
                #pragma omp critical
                {
                    fprintf(stderr, "为DPU %u构建稀疏映射失败\n", dpu_idx);
                    build_failed = 1;
                }
            }
        }
        
        if (build_failed) {
            for (uint32_t j = 0; j < num_dpus; ++j) {
                sparse_comm_map_free(&dpu_maps[j]);
            }
            free(dpu_maps);
            return UINT64_MAX;
        }
        
        double build_end = profile_time_now();
        batch_prep_time += (build_end - build_start);
        
        // === 阶段2: 传输稀疏映射和批次数据到DPU（优化版） ===
        double transfer_start = profile_time_now();
        
        // 统计传输数据量
        size_t broadcast_bytes = 3 * sizeof(uint32_t);  // batch参数
        size_t sparse_map_bytes = 0;
        
        // 2.1 Broadcast批次参数（所有DPU相同）- 使用异步传输
        if (dpu_broadcast_to(dpu_set, "batch_start_idx", 0, &batch_start, sizeof(uint32_t), DPU_XFER_ASYNC) != DPU_OK) {
            fprintf(stderr, "Broadcast batch_start_idx失败\n");
            for (uint32_t j = 0; j < num_dpus; ++j) {
                sparse_comm_map_free(&dpu_maps[j]);
            }
            free(dpu_maps);
            return UINT64_MAX;
        }
        if (dpu_broadcast_to(dpu_set, "batch_count", 0, &batch_count, sizeof(uint32_t), DPU_XFER_ASYNC) != DPU_OK) {
            fprintf(stderr, "Broadcast batch_count失败\n");
            for (uint32_t j = 0; j < num_dpus; ++j) {
                sparse_comm_map_free(&dpu_maps[j]);
            }
            free(dpu_maps);
            return UINT64_MAX;
        }
        if (dpu_broadcast_to(dpu_set, "total_batches", 0, &num_batches, sizeof(uint32_t), DPU_XFER_DEFAULT) != DPU_OK) {
            fprintf(stderr, "Broadcast total_batches失败\n");
            for (uint32_t j = 0; j < num_dpus; ++j) {
                sparse_comm_map_free(&dpu_maps[j]);
            }
            free(dpu_maps);
            return UINT64_MAX;
        }
        
        // 2.2 Broadcast批次顶点列表（所有DPU相同）
        if (batch_count > 0) {
            size_t vertex_bytes = batch_count * sizeof(uint32_t);
            broadcast_bytes += vertex_bytes;
            size_t padded_size = ((vertex_bytes + 7) / 8) * 8;
            if (dpu_broadcast_to(dpu_set, "delta_vertex_ids", 0,
                               vertices + batch_start,
                               padded_size, DPU_XFER_DEFAULT) != DPU_OK) {
                fprintf(stderr, "Broadcast顶点批次失败\n");
                for (uint32_t j = 0; j < num_dpus; ++j) {
                    sparse_comm_map_free(&dpu_maps[j]);
                }
                free(dpu_maps);
                return UINT64_MAX;
            }
        }
        
        // 2.3 并行传输每个DPU的稀疏映射
        struct dpu_set_t dpu;
        uint32_t dpu_idx = 0;
        
        // 计算总的稀疏映射数据量
        for (uint32_t i = 0; i < num_dpus; ++i) {
            const sparse_comm_map_t *map = &dpu_maps[i];
            sparse_map_bytes += 2 * sizeof(uint32_t);  // 两个size字段
            sparse_map_bytes += map->neighbor_count * 2 * sizeof(uint32_t);  // neighbor_ids + neighbor_comms
            sparse_map_bytes += map->comm_count * 2 * sizeof(uint32_t);      // comm_ids + comm_degrees
        }
        
        // 传输每个DPU的稀疏映射
        DPU_FOREACH(dpu_set, dpu) {
            if (dpu_idx >= num_dpus) break;
            
            const sparse_comm_map_t *map = &dpu_maps[dpu_idx];
            
            // 传输邻居映射大小和数据
            if (dpu_copy_to(dpu, "neighbor_map_size", 0,
                           &map->neighbor_count, sizeof(uint32_t)) != DPU_OK) {
                fprintf(stderr, "传输neighbor_map_size到DPU %u失败\n", dpu_idx);
                for (uint32_t j = 0; j < num_dpus; ++j) {
                    sparse_comm_map_free(&dpu_maps[j]);
                }
                free(dpu_maps);
                return UINT64_MAX;
            }
            
            if (map->neighbor_count > 0) {
                // MRAM要求8字节对齐，计算对齐后的大小
                size_t neighbor_bytes = map->neighbor_count * sizeof(uint32_t);
                size_t aligned_bytes = ((neighbor_bytes + 7) / 8) * 8;
                
                // 传输neighbor_ids（使用对齐大小）
                if (dpu_copy_to(dpu, "neighbor_ids", 0,
                               map->neighbor_ids,
                               aligned_bytes) != DPU_OK) {
                    fprintf(stderr, "传输neighbor_ids到DPU %u失败\n", dpu_idx);
                    for (uint32_t j = 0; j < num_dpus; ++j) {
                        sparse_comm_map_free(&dpu_maps[j]);
                    }
                    free(dpu_maps);
                    return UINT64_MAX;
                }
                
                // 传输neighbor_comms（使用对齐大小）
                if (dpu_copy_to(dpu, "neighbor_comms", 0,
                               map->neighbor_comms,
                               aligned_bytes) != DPU_OK) {
                    fprintf(stderr, "传输neighbor_comms到DPU %u失败\n", dpu_idx);
                    for (uint32_t j = 0; j < num_dpus; ++j) {
                        sparse_comm_map_free(&dpu_maps[j]);
                    }
                    free(dpu_maps);
                    return UINT64_MAX;
                }
            }
            
            // 传输社区度数映射大小和数据
            if (dpu_copy_to(dpu, "comm_map_size", 0,
                           &map->comm_count, sizeof(uint32_t)) != DPU_OK) {
                fprintf(stderr, "传输comm_map_size到DPU %u失败\n", dpu_idx);
                for (uint32_t j = 0; j < num_dpus; ++j) {
                    sparse_comm_map_free(&dpu_maps[j]);
                }
                free(dpu_maps);
                return UINT64_MAX;
            }
            
            if (map->comm_count > 0) {
                // MRAM要求8字节对齐，计算对齐后的大小
                size_t comm_bytes = map->comm_count * sizeof(uint32_t);
                size_t aligned_comm_bytes = ((comm_bytes + 7) / 8) * 8;
                
                // 传输comm_ids（使用对齐大小）
                if (dpu_copy_to(dpu, "comm_ids", 0,
                               map->comm_ids,
                               aligned_comm_bytes) != DPU_OK) {
                    fprintf(stderr, "传输comm_ids到DPU %u失败\n", dpu_idx);
                    for (uint32_t j = 0; j < num_dpus; ++j) {
                        sparse_comm_map_free(&dpu_maps[j]);
                    }
                    free(dpu_maps);
                    return UINT64_MAX;
                }
                
                // 传输comm_degrees（使用对齐大小）
                if (dpu_copy_to(dpu, "comm_degrees", 0,
                               map->comm_degrees,
                               aligned_comm_bytes) != DPU_OK) {
                    fprintf(stderr, "传输comm_degrees到DPU %u失败\n", dpu_idx);
                    for (uint32_t j = 0; j < num_dpus; ++j) {
                        sparse_comm_map_free(&dpu_maps[j]);
                    }
                    free(dpu_maps);
                    return UINT64_MAX;
                }
            }
            
            dpu_idx++;
        }
        
        // 释放稀疏映射（已传输到DPU）
        for (uint32_t j = 0; j < num_dpus; ++j) {
            sparse_comm_map_free(&dpu_maps[j]);
        }
        free(dpu_maps);
        
        double transfer_end = profile_time_now();
        g_profile.host_to_dpu_params += (transfer_end - transfer_start);
        
        // === 阶段3: 设置phase和two_m ===
        double phase_start = profile_time_now();
        if (!broadcast_phase(dpu_set, phase, two_m)) {
            return UINT64_MAX;
        }
        double phase_end = profile_time_now();
        g_profile.host_to_dpu_params += (phase_end - phase_start);
        
        // === 阶段4: 执行DPU计算 ===
        uint64_t batch_moves = run_phase(dpu_set,
                                        changed_nodes_out,
                                        changed_comms_out,
                                        changed_old_comms_out,
                                        changed_degrees_out,
                                        dpu_time_accum);
        
        if (batch_moves == UINT64_MAX) {
            return UINT64_MAX;
        }
        
        total_moves += batch_moves;
        profile_printf("      批次 %u 移动数: %" PRIu64 " [传输: broadcast %.2f MB + sparse %.2f MB]\n",
                      batch + 1, batch_moves,
                      broadcast_bytes / (1024.0 * 1024.0),
                      sparse_map_bytes / (1024.0 * 1024.0));
    }
    
    // 将批次准备时间计入broadcast_updates（稀疏映射构建+传输）
    g_profile.broadcast_updates += batch_prep_time;
    
    return total_moves;
}

// 主函数 - 基于原host.c修改
int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    
    const char *filename = argv[1];
    uint32_t num_dpus = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1;
    if (num_dpus == 0) num_dpus = 1;
    
    uint32_t max_local_iters = (argc >= 4) ? (uint32_t)atoi(argv[3]) : MAX_LOCAL_ITERS;
    if (max_local_iters == 0) max_local_iters = 1;
    
    const char *output_csv = (argc >= 5) ? argv[4] : "upmem_membership.csv";
    const char *membership_path = (argc >= 6) ? argv[5] : NULL;

    // 初始化随机数种子
    srand((unsigned int)time(NULL));

    graph_t graph;
    if (!load_graph(filename, &graph)) {
        return 1;
    }
    g_row_ptr_for_sort = graph.row_ptr;
    const uint32_t original_nodes = graph.n_nodes;

    uint32_t *node_degree = compute_degrees(&graph);
    if (!node_degree) {
        fprintf(stderr, "计算节点度数失败\n");
        free_graph(&graph);
        return 1;
    }

    uint32_t *node_comm = NULL;
    uint32_t *orig_to_current = malloc(original_nodes * sizeof(uint32_t));
    if (!orig_to_current) {
        fprintf(stderr, "分配orig_to_current数组失败\n");
        free(node_degree);
        free_graph(&graph);
        return 1;
    }
    for (uint32_t i = 0; i < original_nodes; ++i) {
        orig_to_current[i] = i;
    }

    uint32_t num_communities = graph.n_nodes;
    community_span_t *comm_spans = NULL;
    uint32_t *dpu_vertex_starts = NULL;
    uint32_t *dpu_vertex_counts = NULL;

    if (membership_path && membership_path[0] != '\0') {
        uint32_t *offline_membership = NULL;
        if (!load_membership_file(membership_path, graph.n_nodes,
                                  &offline_membership, &num_communities)) {
            fprintf(stderr, "加载membership文件 '%s' 失败\n", membership_path);
            free(node_degree);
            free_graph(&graph);
            free(orig_to_current);
            return 1;
        }
        
        comm_spans = reorder_graph_by_membership(&graph, &node_degree, &node_comm,
                                                 offline_membership, num_communities);
        free(offline_membership);
        if (!comm_spans) {
            fprintf(stderr, "使用离线membership重排图失败\n");
            free(node_degree);
            free_graph(&graph);
            free(orig_to_current);
            return 1;
        }
        g_row_ptr_for_sort = graph.row_ptr;
        
        if (!assign_communities_to_dpus(comm_spans, num_communities, graph.n_nodes,
                                       num_dpus, &g_vertex_owner, &g_community_owner,
                                       &dpu_vertex_starts, &dpu_vertex_counts)) {
            fprintf(stderr, "映射社区到DPU失败\n");
            free(node_degree);
            free_graph(&graph);
            free(orig_to_current);
            free(comm_spans);
            reset_global_ownership();
            return 1;
        }
        g_num_communities = num_communities;
    } else {
        // 没有membership文件，每个节点独立成社区
        node_comm = malloc(graph.n_nodes * sizeof(uint32_t));
        if (!node_comm) {
            fprintf(stderr, "分配node_comm数组失败\n");
            free(node_degree);
            free_graph(&graph);
            free(orig_to_current);
            return 1;
        }
        for (uint32_t i = 0; i < graph.n_nodes; ++i) {
            node_comm[i] = i;
        }
        g_num_communities = graph.n_nodes;
    }
    
    // 在图重排之后保存原始图和度数（用于计算模块度）
    graph_t original_graph;
    original_graph.n_nodes = graph.n_nodes;
    original_graph.n_edges = graph.n_edges;
    original_graph.row_ptr = malloc((graph.n_nodes + 1) * sizeof(uint32_t));
    original_graph.col_idx = malloc(graph.n_edges * sizeof(uint32_t));
    original_graph.new_to_old = NULL;
    
    if (!original_graph.row_ptr || !original_graph.col_idx) {
        fprintf(stderr, "无法分配原始图副本\n");
        free(original_graph.row_ptr);
        free(original_graph.col_idx);
        free(node_comm);
        free(node_degree);
        free_graph(&graph);
        free(orig_to_current);
        free(comm_spans);
        free(dpu_vertex_starts);
        free(dpu_vertex_counts);
        reset_global_ownership();
        return 1;
    }
    
    memcpy(original_graph.row_ptr, graph.row_ptr, (graph.n_nodes + 1) * sizeof(uint32_t));
    memcpy(original_graph.col_idx, graph.col_idx, graph.n_edges * sizeof(uint32_t));
    
    // 保存原始度数和社区分配
    uint32_t *original_node_degree = malloc(original_nodes * sizeof(uint32_t));
    uint32_t *original_membership = malloc(original_nodes * sizeof(uint32_t));
    if (!original_node_degree || !original_membership) {
        fprintf(stderr, "分配原始度数/membership数组失败\n");
        free(original_node_degree);
        free(original_membership);
        free_graph(&original_graph);
        free(node_comm);
        free(node_degree);
        free_graph(&graph);
        free(orig_to_current);
        free(comm_spans);
        free(dpu_vertex_starts);
        free(dpu_vertex_counts);
        reset_global_ownership();
        return 1;
    }
    memcpy(original_node_degree, node_degree, original_nodes * sizeof(uint32_t));
    memcpy(original_membership, node_comm, original_nodes * sizeof(uint32_t));

    uint32_t *comm_tot_degree = malloc(graph.n_nodes * sizeof(uint32_t));
    if (!comm_tot_degree) {
        fprintf(stderr, "分配comm_tot_degree数组失败\n");
        free(original_membership);
        free(original_node_degree);
        free_graph(&original_graph);
        free(node_comm);
        free(orig_to_current);
        free(node_degree);
        free_graph(&graph);
        reset_global_ownership();
        return 1;
    }

    if (!compute_comm_totals(&graph, node_degree, node_comm, comm_tot_degree)) {
        fprintf(stderr, "计算初始社区总度数失败\n");
        free(original_membership);
        free(original_node_degree);
        free_graph(&original_graph);
        free(node_comm);
        free(orig_to_current);
        free(node_degree);
        free_graph(&graph);
        free(comm_tot_degree);
        reset_global_ownership();
        return 1;
    }

    struct dpu_set_t dpu_set;
    if (dpu_alloc(num_dpus, NULL, &dpu_set) != DPU_OK) {
        fprintf(stderr, "DPU分配失败\n");
        free(original_membership);
        free(original_node_degree);
        free_graph(&original_graph);
        free(node_comm);
        free(orig_to_current);
        free(node_degree);
        free_graph(&graph);
        free(comm_spans);
        free(dpu_vertex_starts);
        free(dpu_vertex_counts);
        free(comm_tot_degree);
        reset_global_ownership();
        return 1;
    }
    
    struct dpu_program_t *program = NULL;
    if (dpu_load(dpu_set, DPU_BINARY, &program) != DPU_OK) {
        fprintf(stderr, "dpu_load失败\n");
        dpu_free(dpu_set);
        free(original_membership);
        free(original_node_degree);
        free_graph(&original_graph);
        free(node_comm);
        free(orig_to_current);
        free(node_degree);
        free_graph(&graph);
        free(comm_spans);
        free(dpu_vertex_starts);
        free(dpu_vertex_counts);
        free(comm_tot_degree);
        reset_global_ownership();
        return 1;
    }

    double partition_start = profile_time_now();
    dpu_partition_t *partitions = setup_partitions(&graph, node_degree, node_comm,
                                                   dpu_set, num_dpus,
                                                   dpu_vertex_starts,
                                                   dpu_vertex_counts);
    g_profile.partition_initial += profile_time_now() - partition_start;
    
    if (!partitions) {
        fprintf(stderr, "设置DPU分区失败\n");
        dpu_free(dpu_set);
        free(original_membership);
        free(original_node_degree);
        free_graph(&original_graph);
        free(node_comm);
        free(orig_to_current);
        free(node_degree);
        free_graph(&graph);
        free(comm_spans);
        free(dpu_vertex_starts);
        free(dpu_vertex_counts);
        free(comm_tot_degree);
        reset_global_ownership();
        return 1;
    }
    free(comm_spans);
    free(dpu_vertex_starts);
    free(dpu_vertex_counts);

    // 获取增量百分比
    const char *delta_env = getenv("DELTA_PERCENT");
    if (delta_env) {
        double val = strtod(delta_env, NULL);
        if (val > 100.0) val = 100.0;
        if (val < 0.0) val = 0.0;
        g_delta_fraction_percent = (uint32_t)(val * 100.0);
        if (g_delta_fraction_percent > 10000) g_delta_fraction_percent = 10000;
        profile_printf("使用增量比例 %.2f%% (环境变量)\n", 
                      (double)g_delta_fraction_percent / 100.0);
    }

    int exit_code = 0;
    bool fatal_error = false;
    uint32_t level = 0;
    double total_dpu_time = 0.0;

    struct timespec algo_start, algo_end;
    clock_gettime(CLOCK_MONOTONIC, &algo_start);
    
    // 计算初始模块度（使用加载的社区分配或每个节点独立）
    double initial_modularity = compute_graph_modularity(&original_graph, original_node_degree, original_membership);
    
    printf("\n========================================\n");
    printf("初始模块度 (Initial Modularity) Q = %.6f\n", initial_modularity);
    printf("========================================\n\n");
    
    while (level < MAX_LEVELS) {
        profile_printf("=== Level %u ===\n", level + 1);
        bool any_moves = false;

        // 生成增量顶点列表
        u32_array_t delta_vertices;
        u32_array_init(&delta_vertices);
        
        if (g_delta_fraction_percent > 0) {
            uint32_t delta_count = (uint32_t)(((uint64_t)graph.n_nodes * g_delta_fraction_percent) / 10000);
            if (delta_count == 0) delta_count = 1;
            if (delta_count > graph.n_nodes) delta_count = graph.n_nodes;
            
            // 随机生成：使用 Fisher-Yates 洗牌算法部分打乱
            // 创建索引数组
            uint32_t *indices = (uint32_t*)malloc(graph.n_nodes * sizeof(uint32_t));
            if (!indices) {
                fprintf(stderr, "分配索引数组失败\n");
                fatal_error = true;
            } else {
                // 初始化索引数组
                for (uint32_t i = 0; i < graph.n_nodes; ++i) {
                    indices[i] = i;
                }
                
                // 部分 Fisher-Yates 洗牌：只打乱前 delta_count 个位置
                for (uint32_t i = 0; i < delta_count; ++i) {
                    uint32_t j = i + (rand() % (graph.n_nodes - i));
                    uint32_t temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
                
                // 将前 delta_count 个节点加入增量列表
                for (uint32_t i = 0; i < delta_count; ++i) {
                    if (!u32_array_push(&delta_vertices, indices[i])) {
                        fprintf(stderr, "分配增量顶点列表失败\n");
                        fatal_error = true;
                        break;
                    }
                }
                
                free(indices);
            }
        } else {
            // 全量模式：所有节点
            for (uint32_t i = 0; i < graph.n_nodes; ++i) {
                if (!u32_array_push(&delta_vertices, i)) {
                    fprintf(stderr, "分配顶点列表失败\n");
                    fatal_error = true;
                    break;
                }
            }
        }
        
        if (fatal_error) {
            u32_array_free(&delta_vertices);
            break;
        }

        for (uint32_t iter = 0; iter < max_local_iters; ++iter) {
            uint64_t total_degree = compute_total_degree(&graph, node_degree);
            uint32_t two_m = (total_degree > UINT32_MAX) ? UINT32_MAX : (uint32_t)total_degree;

            u32_array_t changed_nodes;
            u32_array_t changed_comms;
            u32_array_t changed_old_comms;
            u32_array_t changed_degrees;
            u32_array_init(&changed_nodes);
            u32_array_init(&changed_comms);
            u32_array_init(&changed_old_comms);
            u32_array_init(&changed_degrees);
            
            // 使用分批处理
            uint64_t moves = run_phase_batched(
                dpu_set,
                partitions,
                num_dpus,
                &graph,
                node_comm,
                comm_tot_degree,
                delta_vertices.data,
                (uint32_t)delta_vertices.size,
                0,  // phase = 0 (local moving)
                two_m,
                &changed_nodes,
                &changed_comms,
                &changed_old_comms,
                &changed_degrees,
                &total_dpu_time
            );
            
            if (moves == UINT64_MAX) {
                fatal_error = true;
                u32_array_free(&changed_nodes);
                u32_array_free(&changed_comms);
                u32_array_free(&changed_old_comms);
                u32_array_free(&changed_degrees);
                break;
            }
            
            profile_printf("  Local moving iteration %u: %" PRIu64 " moves\n", iter + 1, moves);
            
            if (moves == 0) {
                u32_array_free(&changed_nodes);
                u32_array_free(&changed_comms);
                u32_array_free(&changed_old_comms);
                u32_array_free(&changed_degrees);
                break;
            }
            
            any_moves = true;

            // 应用变更
            u32_array_t immediate_comm_ids;
            u32_array_t immediate_comm_vals;
            u32_array_t immediate_tot_ids;
            u32_array_t immediate_tot_vals;
            u32_array_init(&immediate_comm_ids);
            u32_array_init(&immediate_comm_vals);
            u32_array_init(&immediate_tot_ids);
            u32_array_init(&immediate_tot_vals);
            
            double merge_start = profile_time_now();
            if (!apply_changes_and_prepare_updates(changed_nodes.data, changed_comms.data,
                                                   changed_old_comms.data, changed_degrees.data,
                                                   changed_nodes.size,
                                                   graph.n_nodes, node_comm, comm_tot_degree,
                                                   &immediate_comm_ids, &immediate_comm_vals,
                                                   &immediate_tot_ids, &immediate_tot_vals)) {
                fatal_error = true;
                u32_array_free(&changed_nodes);
                u32_array_free(&changed_comms);
                u32_array_free(&changed_old_comms);
                u32_array_free(&changed_degrees);
                u32_array_free(&immediate_comm_ids);
                u32_array_free(&immediate_comm_vals);
                u32_array_free(&immediate_tot_ids);
                u32_array_free(&immediate_tot_vals);
                break;
            }
            double merge_end = profile_time_now();
            g_profile.merge_changes += (merge_end - merge_start);

            u32_array_free(&changed_nodes);
            u32_array_free(&changed_comms);
            u32_array_free(&changed_old_comms);
            u32_array_free(&changed_degrees);
            u32_array_free(&immediate_comm_ids);
            u32_array_free(&immediate_comm_vals);
            u32_array_free(&immediate_tot_ids);
            u32_array_free(&immediate_tot_vals);
            
            if (fatal_error) break;
        }

        u32_array_free(&delta_vertices);
        
        if (fatal_error) break;

        if (!any_moves) {
            profile_printf("本层无移动；停止。\n");
            break;
        }

        // 图收缩
        double contraction_start = profile_time_now();
        int aggr_status = aggregate_graph(&graph, &node_degree, &node_comm, orig_to_current, original_nodes);
        if (aggr_status < 0) {
            g_profile.contraction += profile_time_now() - contraction_start;
            fprintf(stderr, "图聚合失败\n");
            fatal_error = true;
            break;
        }
        if (aggr_status == 0) {
            g_profile.contraction += profile_time_now() - contraction_start;
            profile_printf("图收缩停滞；完成。\n");
            break;
        }

        g_row_ptr_for_sort = graph.row_ptr;
        free(comm_tot_degree);
        comm_tot_degree = malloc(graph.n_nodes * sizeof(uint32_t));
        if (!comm_tot_degree || !compute_comm_totals(&graph, node_degree, node_comm, comm_tot_degree)) {
            g_profile.contraction += profile_time_now() - contraction_start;
            fprintf(stderr, "聚合后重新计算社区总度数失败\n");
            fatal_error = true;
            break;
        }
        
        reset_global_ownership();
        free_partitions(partitions, num_dpus);
        partitions = setup_partitions(&graph, node_degree, node_comm,
                                     dpu_set, num_dpus, NULL, NULL);
        if (!partitions) {
            g_profile.contraction += profile_time_now() - contraction_start;
            fprintf(stderr, "聚合后重建分区失败\n");
            fatal_error = true;
            break;
        }

        g_profile.contraction += profile_time_now() - contraction_start;
        level++;
    }

    if (fatal_error) {
        exit_code = 1;
    } else {
        uint32_t *final_membership = malloc(original_nodes * sizeof(uint32_t));
        if (final_membership) {
            for (uint32_t i = 0; i < original_nodes; ++i) {
                uint32_t curr = orig_to_current[i];
                if (curr >= graph.n_nodes) {
                    curr = graph.n_nodes - 1;
                }
                final_membership[i] = node_comm[curr];
            }
            
            uint32_t distinct = 0;
            uint8_t *seen = calloc(original_nodes, sizeof(uint8_t));
            if (seen) {
                for (uint32_t i = 0; i < original_nodes; ++i) {
                    uint32_t cid = final_membership[i];
                    if (cid < original_nodes && !seen[cid]) {
                        seen[cid] = 1;
                        distinct++;
                    }
                }
                free(seen);
            }
            profile_printf("检测到 %u 个社区，共 %u 个原始节点。\n", distinct, original_nodes);

            double csv_start = profile_time_now();
            FILE *csv = fopen(output_csv, "w");
            if (!csv) {
                fprintf(stderr, "警告: 无法打开 %s 进行写入\n", output_csv);
            } else {
                for (uint32_t i = 0; i < original_nodes; ++i) {
                    uint32_t node_id = graph.new_to_old ? graph.new_to_old[i] : i;
                    fprintf(csv, "%u,%u\n", node_id, final_membership[i]);
                }
                fclose(csv);
                profile_printf("社区分配已写入 %s\n", output_csv);
            }
            g_profile.csv_output += profile_time_now() - csv_start;
            
            // 计算最终模块度（用原始图 + 最终社区分配）
            double final_modularity = compute_graph_modularity(&original_graph, original_node_degree, final_membership);
            
            double modularity_improvement = final_modularity - initial_modularity;
            double improvement_percent = (initial_modularity != 0.0) ? 
                (modularity_improvement / fabs(initial_modularity) * 100.0) : 0.0;
            
            printf("\n========================================\n");
            printf("模块度对比 (Modularity Comparison)\n");
            printf("========================================\n");
            printf("初始模块度: Q₀ = %.6f\n", initial_modularity);
            printf("最终模块度: Q  = %.6f\n", final_modularity);
            printf("模块度提升: ΔQ = %.6f", modularity_improvement);
            if (modularity_improvement > 0) {
                printf(" ↑ (+%.2f%%)\n", improvement_percent);
            } else if (modularity_improvement < 0) {
                printf(" ↓ (%.2f%%)\n", improvement_percent);
            } else {
                printf(" (无变化)\n");
            }
            printf("========================================\n");

            free(final_membership);
        } else {
            fprintf(stderr, "警告: 无法分配最终membership缓冲区\n");
        }
    }

    free(comm_tot_degree);
    free_partitions(partitions, num_dpus);
    dpu_free(dpu_set);
    free(node_comm);
    free(orig_to_current);
    free(node_degree);
    free(original_membership);    // 释放原始社区分配数组
    free(original_node_degree);   // 释放原始度数数组
    free_graph(&graph);
    free_graph(&original_graph);  // 释放原始图副本
    reset_global_ownership();

    clock_gettime(CLOCK_MONOTONIC, &algo_end);
    double total_duration = elapsed_seconds(&algo_start, &algo_end);

    double total_A = g_profile.load_read_edges + g_profile.load_build_csr;
    double total_B = g_profile.partition_initial + g_profile.compute_totals;
    double total_C = g_profile.broadcast_updates;
    double total_D = g_profile.host_to_dpu_params;
    double total_E = g_profile.dpu_launch;
    double total_F = g_profile.dpu_fetch_changes;
    double total_G = g_profile.merge_changes;
    double total_H = g_profile.logging + g_profile.csv_output;
    double total_I = g_profile.contraction;
    double total_J = g_profile.compute_modularity;

    printf("\n=== Host Profiling (seconds) ===\n");
    profile_printf("总运行时间: %.3f s (DPU内核 %.3f s)\n", total_duration - total_H, total_dpu_time);
    printf("**[A]** Load edges/CSR build: %.6f (read %.6f + build %.6f)\n",
           total_A, g_profile.load_read_edges, g_profile.load_build_csr);
    printf("**[B]** Initial partition & comm totals: %.6f (partition %.6f + totals %.6f)\n",
           total_B, g_profile.partition_initial, g_profile.compute_totals);
    printf("**[C]** Build sparse comm maps (batch prep): %.6f\n", total_C);
    printf("**[D]** Transfer maps + params to DPU: %.6f\n", total_D);
    printf("**[E]** DPU kernel execution: %.6f\n", total_E);
    printf("**[F]** Fetch results from DPU: %.6f\n", total_F);
    printf("**[G]** Merge community updates: %.6f\n", total_G);
    printf("**[H]** Logging / CSV output: %.6f (stdout %.6f + csv %.6f)\n",
           total_H, g_profile.logging, g_profile.csv_output);
    printf("**[I]** Graph contraction & repartition: %.6f\n", total_I);
    printf("**[J]** Modularity evaluation: %.6f\n", total_J);

    return exit_code;
}

