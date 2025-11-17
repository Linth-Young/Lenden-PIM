#ifndef HOST_DPU_COMM_H
#define HOST_DPU_COMM_H

#include <dpu.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifndef MAX_NODES_PER_DPU
#define MAX_NODES_PER_DPU 600000
#endif
#ifndef MAX_EDGES_PER_DPU
#define MAX_EDGES_PER_DPU 1200000
#endif
#ifndef MAX_GLOBAL_NODES
#define MAX_GLOBAL_NODES 60000000
#endif
#ifndef MAX_COMM_UPDATES
#define MAX_COMM_UPDATES (2 * MAX_GLOBAL_NODES)
#endif
#ifndef MAX_INCREMENTAL_ITEMS
#define MAX_INCREMENTAL_ITEMS MAX_COMM_UPDATES
#endif
#ifndef MAX_UPDATE_BATCH
#define MAX_UPDATE_BATCH 65536u
#endif
#ifndef MAX_DELTA_ITEMS
#define MAX_DELTA_ITEMS MAX_UPDATE_BATCH
#endif

#define SYMBOL_INC_VERTEX_IDS   "delta_vertex_ids"
#define SYMBOL_INC_VERTEX_COUNT "delta_vertex_count"

#define MAX_LOCAL_ITERS 10
#define MAX_LEVELS 5

typedef struct {
    double load_read_edges;
    double load_build_csr;
    double partition_initial;
    double compute_totals;
    double compute_modularity;
    double broadcast_updates;
    double host_to_dpu_params;
    double dpu_launch;
    double dpu_fetch_changes;
    double merge_changes;
    double contraction;
    double logging;
    double csv_output;
} host_profile_t;

extern host_profile_t g_profile;
extern uint32_t *g_vertex_owner;
extern uint32_t *g_community_owner;
extern uint32_t g_num_communities;
// 改为整数百分比：0-10000 表示 0%-100.00%
extern uint32_t g_delta_fraction_percent;

uint64_t prng_step(uint64_t *state);
void reset_global_ownership(void);
double profile_time_now(void);
void profile_printf(const char *fmt, ...);

typedef struct {
    uint32_t *data;
    size_t size;
    size_t capacity;
} u32_array_t;

void u32_array_init(u32_array_t *arr);
void u32_array_clear(u32_array_t *arr);
void u32_array_free(u32_array_t *arr);
bool u32_array_reserve(u32_array_t *arr, size_t new_cap);
bool u32_array_extend(u32_array_t *arr, size_t add_count, uint32_t **out_ptr);
bool u32_array_push(u32_array_t *arr, uint32_t value);

extern const uint32_t *g_row_ptr_for_sort;
int compare_by_rowptr(const void *a, const void *b);

void usage(const char *prog);

typedef struct {
    uint32_t n_nodes;
    uint32_t n_edges;
    uint32_t *row_ptr;
    uint32_t *col_idx;
    uint32_t *new_to_old;
} graph_t;

void free_graph(graph_t *graph);

typedef struct {
    uint32_t start_node;
    uint32_t local_nodes;
    uint32_t start_edge;
    uint32_t edge_count;
    uint32_t *node_degree_local;
} dpu_partition_t;

typedef struct {
    uint32_t id;
    int64_t delta;
} comm_delta_t;

typedef struct {
    uint32_t id;
    uint32_t size;
    uint64_t edge_load;
    uint32_t offset;
} community_block_t;

typedef struct {
    uint32_t new_id;
    uint32_t original_id;
    uint32_t start;
    uint32_t length;
    uint64_t degree_sum;
} community_span_t;

typedef enum {
    EDGE_INSERTION = 0,
    EDGE_DELETION = 1
} edge_update_type_t;

typedef struct {
    edge_update_type_t type;
    uint32_t src;
    uint32_t dst;
    uint32_t weight;
} edge_update_t;

typedef struct {
    FILE *fp;
    char *path;
    uint32_t max_vertex_id;
    uint64_t lines_read;
    bool reached_eof;
} trace_reader_t;

bool trace_reader_open(trace_reader_t *reader, const char *path, uint32_t max_vertex_id);
size_t trace_reader_next_batch(trace_reader_t *reader, edge_update_t *buffer, size_t capacity);
void trace_reader_close(trace_reader_t *reader);

typedef struct {
    u32_array_t vertices;
    u32_array_t communities;
    u32_array_t candidate_comms;
    u32_array_t update_indices;
} dpu_incremental_work_t;

typedef struct {
    u32_array_t comm_ids;
    u32_array_t target_dpus;
} remote_notification_t;

typedef struct {
    dpu_incremental_work_t *worklists;
    remote_notification_t *notifications;
    uint32_t num_dpus;
} incremental_context_t;

dpu_error_t dpu_copy_padded(struct dpu_set_t dpu,
                            const char *symbol,
                            uint32_t offset,
                            const void *src,
                            uint32_t size);
dpu_error_t dpu_copy_from_padded(struct dpu_set_t dpu,
                                 const char *symbol,
                                 uint32_t offset,
                                 void *dst,
                                 uint32_t size);
double elapsed_seconds(const struct timespec *start, const struct timespec *end);
bool broadcast_u32_array(struct dpu_set_t dpu_set,
                         const char *symbol,
                         const uint32_t *values,
                         uint32_t count);
bool broadcast_updates(struct dpu_set_t dpu_set,
                       const char *ids_symbol,
                       const char *vals_symbol,
                       const char *count_symbol,
                       const uint32_t *ids,
                       const uint32_t *vals,
                       uint32_t count);

bool incremental_context_init(incremental_context_t *ctx, uint32_t num_dpus);
void incremental_context_free(incremental_context_t *ctx);

bool apply_changes_and_prepare_updates(const uint32_t *changed_nodes,
                                       const uint32_t *changed_new_comms,
                                       const uint32_t *changed_old_comms,
                                       const uint32_t *changed_degrees,
                                       size_t change_count,
                                       uint32_t total_nodes,
                                       uint32_t *node_comm,
                                       uint32_t *comm_tot_degree,
                                       u32_array_t *out_comm_ids,
                                       u32_array_t *out_comm_vals,
                                       u32_array_t *out_tot_ids,
                                       u32_array_t *out_tot_vals);

bool load_membership_file(const char *path,
                          uint32_t expected_vertices,
                          uint32_t **out_membership,
                          uint32_t *out_num_communities);
community_span_t *reorder_graph_by_membership(graph_t *graph,
                                               uint32_t **node_degree,
                                               uint32_t **node_comm,
                                               const uint32_t *membership,
                                               uint32_t num_communities);
bool assign_communities_to_dpus(const community_span_t *spans,
                                uint32_t num_communities,
                                uint32_t total_vertices,
                                uint32_t num_dpus,
                                uint32_t **out_vertex_owner,
                                uint32_t **out_community_owner,
                                uint32_t **out_vertex_starts,
                                uint32_t **out_vertex_counts);

bool process_incremental_batch(const edge_update_t *updates,
                               size_t update_count,
                               const uint32_t *current_membership,
                               uint32_t total_vertices,
                               incremental_context_t *ctx);
size_t synthesize_random_updates(edge_update_t *buffer,
                                 size_t capacity,
                                 uint32_t num_vertices,
                                 uint64_t epoch,
                                 uint32_t fraction_percent);
bool broadcast_incremental_worklists_to_dpus(struct dpu_set_t dpu_set,
                                             incremental_context_t *ctx);

bool load_graph(const char *filename, graph_t *graph);
uint32_t *compute_degrees(const graph_t *graph);
void free_partitions(dpu_partition_t *partitions, uint32_t num_dpus);
dpu_partition_t *setup_partitions(const graph_t *graph,
                                  const uint32_t *node_degree,
                                  const uint32_t *node_comm,
                                  struct dpu_set_t dpu_set,
                                  uint32_t num_dpus,
                                  const uint32_t *vertex_starts,
                                  const uint32_t *vertex_counts);
bool compute_comm_totals(const graph_t *graph,
                         const uint32_t *node_degree,
                         const uint32_t *node_comm,
                         uint32_t *comm_tot_degree);
bool broadcast_phase(struct dpu_set_t dpu_set, uint32_t phase, uint32_t two_m);
uint64_t run_phase(struct dpu_set_t dpu_set,
                   u32_array_t *changed_nodes,
                   u32_array_t *changed_comms,
                   u32_array_t *changed_old_comms,
                   u32_array_t *changed_degrees,
                   double *total_dpu_time);
uint64_t compute_total_degree(const graph_t *graph, const uint32_t *node_degree);
double compute_graph_modularity(const graph_t *graph,
                                const uint32_t *node_degree,
                                const uint32_t *node_comm);
int aggregate_graph(graph_t *graph,
                    uint32_t **node_degree,
                    uint32_t **node_comm,
                    uint32_t *orig_to_current,
                    uint32_t original_nodes);

#endif /* HOST_DPU_COMM_H */
