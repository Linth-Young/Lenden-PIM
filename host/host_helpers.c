/*
 * Leiden algorithm orchestration on UPMEM DPUs.
 * Host handles graph loading, level-by-level contraction, and metadata,
 * while DPUs perform bandwidth-heavy local moving/refinement steps.
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
 #include <time.h>
 #include <ctype.h>
 
 #include "host_helpers.h"
 
host_profile_t g_profile;
uint32_t *g_vertex_owner = NULL;
uint32_t *g_community_owner = NULL;
uint32_t g_num_communities = 0;
// 整数百分比：0-10000 表示 0%-100.00% (0 => legacy behavior)
uint32_t g_delta_fraction_percent = 0;
 
 uint64_t prng_step(uint64_t *state) {
     uint64_t x = *state;
     x ^= x >> 12;
     x ^= x << 25;
     x ^= x >> 27;
     *state = x;
     return x * 2685821657736338717ULL;
 }
 
 void reset_global_ownership(void) {
     free(g_vertex_owner);
     free(g_community_owner);
     g_vertex_owner = NULL;
     g_community_owner = NULL;
     g_num_communities = 0;
 }
 
 double profile_time_now(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
 }
 
 void profile_printf(const char *fmt, ...) {
     double start = profile_time_now();
     va_list args;
     va_start(args, fmt);
     vprintf(fmt, args);
     va_end(args);
     g_profile.logging += profile_time_now() - start;
 }
 
 void u32_array_init(u32_array_t *arr) {
     arr->data = NULL;
     arr->size = 0;
     arr->capacity = 0;
 }
 
 void u32_array_clear(u32_array_t *arr) {
     arr->size = 0;
 }
 
 void u32_array_free(u32_array_t *arr) {
     free(arr->data);
     arr->data = NULL;
     arr->size = 0;
     arr->capacity = 0;
 }
 
 bool u32_array_reserve(u32_array_t *arr, size_t new_cap) {
     if (new_cap <= arr->capacity) {
         return true;
     }
     size_t capacity = arr->capacity ? arr->capacity : 1;
     while (capacity < new_cap) {
         capacity *= 2;
     }
     uint32_t *new_data = realloc(arr->data, capacity * sizeof(uint32_t));
     if (!new_data) {
         return false;
     }
     arr->data = new_data;
     arr->capacity = capacity;
     return true;
 }
 
 bool u32_array_extend(u32_array_t *arr, size_t add_count, uint32_t **out_ptr) {
     size_t old_size = arr->size;
     if (!u32_array_reserve(arr, old_size + add_count)) {
         return false;
     }
     arr->size = old_size + add_count;
     *out_ptr = arr->data + old_size;
     return true;
 }
 
 bool u32_array_push(u32_array_t *arr, uint32_t value) {
     uint32_t *slot = NULL;
     if (!u32_array_extend(arr, 1, &slot)) {
         return false;
     }
     *slot = value;
     return true;
 }
 
 static int compare_u32(const void *a, const void *b) {
     uint32_t ua = *(const uint32_t *)a;
     uint32_t ub = *(const uint32_t *)b;
     if (ua < ub) return -1;
     if (ua > ub) return 1;
     return 0;
 }
 
 /* For sorting incremental vertex IDs by CSR row_ptr offsets */
 const uint32_t *g_row_ptr_for_sort = NULL;
 
 int compare_by_rowptr(const void *a, const void *b) {
     uint32_t va = *(const uint32_t *)a;
     uint32_t vb = *(const uint32_t *)b;
     uint32_t ra = g_row_ptr_for_sort ? g_row_ptr_for_sort[va] : va;
     uint32_t rb = g_row_ptr_for_sort ? g_row_ptr_for_sort[vb] : vb;
     if (ra < rb) return -1;
     if (ra > rb) return 1;
     return (va < vb) ? -1 : (va > vb);
 }
 
 void usage(const char *prog) {
     fprintf(stderr,
             "Usage: %s <graph_file> [num_dpus] [max_local_iters] [output_csv] [membership_file] [trace_file]\n",
             prog);
 }
 
 static char *dup_string(const char *src) {
     size_t len = strlen(src);
     char *dst = malloc(len + 1);
     if (!dst) {
         return NULL;
     }
     memcpy(dst, src, len + 1);
     return dst;
 }
 
 bool trace_reader_open(trace_reader_t *reader, const char *path, uint32_t max_vertex_id) {
     if (!reader || !path || path[0] == '\0') {
         return false;
     }
     memset(reader, 0, sizeof(*reader));
     FILE *fp = fopen(path, "r");
     if (!fp) {
         perror("Failed to open trace file");
         return false;
     }
     char *dup = dup_string(path);
     if (!dup) {
         fclose(fp);
         fprintf(stderr, "Failed to allocate trace path buffer\n");
         return false;
     }
     reader->fp = fp;
     reader->path = dup;
     reader->max_vertex_id = max_vertex_id;
     reader->lines_read = 0;
     reader->reached_eof = false;
     return true;
 }
 
 void trace_reader_close(trace_reader_t *reader) {
     if (!reader) {
         return;
     }
     if (reader->fp) {
         fclose(reader->fp);
         reader->fp = NULL;
     }
     free(reader->path);
     reader->path = NULL;
     reader->max_vertex_id = 0;
     reader->lines_read = 0;
     reader->reached_eof = false;
 }
 
 size_t trace_reader_next_batch(trace_reader_t *reader, edge_update_t *buffer, size_t capacity) {
     if (!reader || !reader->fp || !buffer || capacity == 0) {
         return 0;
     }
     size_t produced = 0;
     char line[256];
     while (produced < capacity) {
         if (!fgets(line, sizeof(line), reader->fp)) {
             reader->reached_eof = true;
             break;
         }
         reader->lines_read++;
         char *cursor = line;
         while (*cursor && isspace((unsigned char)*cursor)) {
             cursor++;
         }
         if (*cursor == '\0' || *cursor == '%' || *cursor == '#') {
             continue;
         }
         for (char *c = cursor; *c; ++c) {
             if (*c == ',' || *c == ';') {
                 *c = ' ';
             }
         }
         char *saveptr = NULL;
         char *token = strtok_r(cursor, " \t\r\n", &saveptr);
         if (!token) {
             continue;
         }
         char *endptr = NULL;
         unsigned long u_ul = strtoul(token, &endptr, 10);
         if (*endptr != '\0') {
             continue;
         }
         token = strtok_r(NULL, " \t\r\n", &saveptr);
         if (!token) {
             continue;
         }
         unsigned long v_ul = strtoul(token, &endptr, 10);
         if (*endptr != '\0') {
             continue;
         }
         token = strtok_r(NULL, " \t\r\n", &saveptr);
         unsigned long w_ul = 1;
         if (token) {
             w_ul = strtoul(token, &endptr, 10);
             if (*endptr != '\0') {
                 w_ul = 1;
             }
         }
         if (u_ul > UINT32_MAX || v_ul > UINT32_MAX) {
             continue;
         }
         uint32_t u = (uint32_t)u_ul;
         uint32_t v = (uint32_t)v_ul;
         if (u >= reader->max_vertex_id || v >= reader->max_vertex_id) {
             continue;
         }
         edge_update_t *slot = &buffer[produced++];
         slot->type = EDGE_INSERTION;
         slot->src = u;
         slot->dst = v;
         slot->weight = (uint32_t)((w_ul == 0 || w_ul > UINT32_MAX) ? 1 : w_ul);
     }
     return produced;
 }
 
 void free_graph(graph_t *graph) {
     if (!graph) {
         return;
     }
     free(graph->row_ptr);
     free(graph->col_idx);
     free(graph->new_to_old);
     graph->row_ptr = NULL;
     graph->col_idx = NULL;
     graph->new_to_old = NULL;
     graph->n_nodes = 0;
     graph->n_edges = 0;
 }
 
 dpu_error_t dpu_copy_padded(struct dpu_set_t dpu,
                             const char *symbol,
                             uint32_t offset,
                             const void *src,
                             uint32_t size) {
     if (size == 0) {
         return dpu_copy_to(dpu, symbol, offset, src, size);
     }
     uint32_t aligned = (size + 7u) & ~7u;
     if (aligned == size) {
         return dpu_copy_to(dpu, symbol, offset, src, size);
     }
     uint8_t *buffer = calloc(1, aligned);
     if (!buffer) {
         return DPU_ERR_SYSTEM;
     }
     memcpy(buffer, src, size);
     dpu_error_t status = dpu_copy_to(dpu, symbol, offset, buffer, aligned);
     free(buffer);
     return status;
 }
 
 dpu_error_t dpu_copy_from_padded(struct dpu_set_t dpu,
                                  const char *symbol,
                                  uint32_t offset,
                                  void *dst,
                                  uint32_t size) {
     if (size == 0) {
         return dpu_copy_from(dpu, symbol, offset, dst, size);
     }
     uint32_t aligned = (size + 7u) & ~7u;
     if (aligned == size) {
         return dpu_copy_from(dpu, symbol, offset, dst, size);
     }
     uint8_t *buffer = malloc(aligned);
     if (!buffer) {
         return DPU_ERR_SYSTEM;
     }
     dpu_error_t status = dpu_copy_from(dpu, symbol, offset, buffer, aligned);
     if (status == DPU_OK) {
         memcpy(dst, buffer, size);
     }
     free(buffer);
     return status;
 }
 
 double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
     return (double)(end->tv_sec - start->tv_sec) +
            (double)(end->tv_nsec - start->tv_nsec) / 1e9;
 }
 
 bool broadcast_u32_array(struct dpu_set_t dpu_set,
                          const char *symbol,
                          const uint32_t *data,
                          uint32_t count) {
     if (count == 0) {
         return true;
     }
     size_t size = (size_t)count * sizeof(uint32_t);
     size_t aligned = (size + 7u) & ~7u;
     if (aligned == size) {
         return dpu_copy_to(dpu_set, symbol, 0, data, (uint32_t)aligned) == DPU_OK;
     }
     uint8_t *buffer = calloc(1, aligned);
     if (!buffer) {
         return false;
     }
     memcpy(buffer, data, size);
     dpu_error_t err = dpu_copy_to(dpu_set, symbol, 0, buffer, (uint32_t)aligned);
     free(buffer);
     return err == DPU_OK;
 }
 
 bool broadcast_updates(struct dpu_set_t dpu_set,
                        const char *ids_symbol,
                        const char *values_symbol,
                        const char *count_symbol,
                        const uint32_t *ids,
                        const uint32_t *values,
                        uint32_t count) {
     double start = profile_time_now();
     bool ok = true;
     if (count > MAX_COMM_UPDATES) {
         fprintf(stderr, "Update vector exceeds MAX_COMM_UPDATES (%u > %u)\n",
                 count, (unsigned)MAX_COMM_UPDATES);
         ok = false;
         goto out;
     }
     if (count > 0) {
         if (!broadcast_u32_array(dpu_set, ids_symbol, ids, count)) {
             fprintf(stderr, "Failed to broadcast %s\n", ids_symbol);
             ok = false;
             goto out;
         }
         if (!broadcast_u32_array(dpu_set, values_symbol, values, count)) {
             fprintf(stderr, "Failed to broadcast %s\n", values_symbol);
             ok = false;
             goto out;
         }
     }
     dpu_error_t err = dpu_copy_to(dpu_set, count_symbol, 0, &count, sizeof(count));
     if (err != DPU_OK) {
         fprintf(stderr, "Failed to broadcast %s: %d\n", count_symbol, (int)err);
         ok = false;
         goto out;
     }
 out:
     g_profile.broadcast_updates += profile_time_now() - start;
     return ok;
 }
 
 static void dpu_incremental_work_init(dpu_incremental_work_t *work) {
     u32_array_init(&work->vertices);
     u32_array_init(&work->communities);
     u32_array_init(&work->candidate_comms);
     u32_array_init(&work->update_indices);
 }
 
 static void dpu_incremental_work_clear(dpu_incremental_work_t *work) {
     u32_array_clear(&work->vertices);
     u32_array_clear(&work->communities);
     u32_array_clear(&work->candidate_comms);
     u32_array_clear(&work->update_indices);
 }
 
 static void dpu_incremental_work_free(dpu_incremental_work_t *work) {
     u32_array_free(&work->vertices);
     u32_array_free(&work->communities);
     u32_array_free(&work->candidate_comms);
     u32_array_free(&work->update_indices);
 }
 
 static void remote_notification_init(remote_notification_t *ntf) {
     u32_array_init(&ntf->comm_ids);
     u32_array_init(&ntf->target_dpus);
 }
 
 static void remote_notification_clear(remote_notification_t *ntf) {
     u32_array_clear(&ntf->comm_ids);
     u32_array_clear(&ntf->target_dpus);
 }
 
 static void remote_notification_free(remote_notification_t *ntf) {
     u32_array_free(&ntf->comm_ids);
     u32_array_free(&ntf->target_dpus);
 }
 
 bool incremental_context_init(incremental_context_t *ctx, uint32_t num_dpus) {
     if (!ctx || num_dpus == 0) {
         return false;
     }
     ctx->num_dpus = num_dpus;
     ctx->worklists = calloc(num_dpus, sizeof(dpu_incremental_work_t));
     ctx->notifications = calloc(num_dpus, sizeof(remote_notification_t));
     if (!ctx->worklists || !ctx->notifications) {
         free(ctx->worklists);
         free(ctx->notifications);
         ctx->worklists = NULL;
         ctx->notifications = NULL;
         ctx->num_dpus = 0;
         return false;
     }
     for (uint32_t d = 0; d < num_dpus; ++d) {
         dpu_incremental_work_init(&ctx->worklists[d]);
         remote_notification_init(&ctx->notifications[d]);
     }
     return true;
 }
 
 static void incremental_context_clear(incremental_context_t *ctx) {
     if (!ctx || ctx->num_dpus == 0) {
         return;
     }
     for (uint32_t d = 0; d < ctx->num_dpus; ++d) {
         dpu_incremental_work_clear(&ctx->worklists[d]);
         remote_notification_clear(&ctx->notifications[d]);
     }
 }
 
 void incremental_context_free(incremental_context_t *ctx) {
     if (!ctx || ctx->num_dpus == 0) {
         return;
     }
     for (uint32_t d = 0; d < ctx->num_dpus; ++d) {
         dpu_incremental_work_free(&ctx->worklists[d]);
         remote_notification_free(&ctx->notifications[d]);
     }
     free(ctx->worklists);
     free(ctx->notifications);
     ctx->worklists = NULL;
     ctx->notifications = NULL;
     ctx->num_dpus = 0;
 }
 
 static int compare_comm_delta(const void *a, const void *b) {
     const comm_delta_t *da = (const comm_delta_t *)a;
     const comm_delta_t *db = (const comm_delta_t *)b;
     if (da->id < db->id) return -1;
     if (da->id > db->id) return 1;
     return 0;
 }
 
 bool apply_changes_and_prepare_updates(const uint32_t *changed_nodes,
                                        const uint32_t *changed_comms,
                                        const uint32_t *changed_old_comms,
                                        const uint32_t *changed_degrees,
                                        size_t change_count,
                                        uint32_t total_nodes,
                                        uint32_t *node_comm,
                                        uint32_t *comm_tot_degree,
                                        u32_array_t *out_comm_ids,
                                        u32_array_t *out_comm_vals,
                                        u32_array_t *out_tot_ids,
                                        u32_array_t *out_tot_vals) {
     double start = profile_time_now();
     bool ok = true;
     u32_array_clear(out_comm_ids);
     u32_array_clear(out_comm_vals);
     u32_array_clear(out_tot_ids);
     u32_array_clear(out_tot_vals);
 
     if (change_count == 0) {
         goto out;
     }
 
     comm_delta_t *deltas = malloc(sizeof(comm_delta_t) * change_count * 2);
     if (!deltas) {
         ok = false;
         goto out;
     }
     size_t delta_count = 0;
 
     for (size_t i = 0; i < change_count; ++i) {
         uint32_t gid = changed_nodes[i];
         if (gid >= total_nodes) {
             continue;
         }
         uint32_t new_comm = changed_comms[i];
         uint32_t degree = changed_degrees ? changed_degrees[i] : 0;
         uint32_t old_comm = changed_old_comms ? changed_old_comms[i] : node_comm[gid];
         if (old_comm == new_comm) {
             continue;
         }
         node_comm[gid] = new_comm;
         if (!u32_array_push(out_comm_ids, gid) || !u32_array_push(out_comm_vals, new_comm)) {
             free(deltas);
             ok = false;
             goto out;
         }
         if (old_comm < total_nodes) {
             deltas[delta_count].id = old_comm;
             deltas[delta_count].delta = -(int64_t)degree;
             delta_count++;
         }
         if (new_comm < total_nodes) {
             deltas[delta_count].id = new_comm;
             deltas[delta_count].delta = (int64_t)degree;
             delta_count++;
         }
     }
 
     if (delta_count > 0) {
         qsort(deltas, delta_count, sizeof(comm_delta_t), compare_comm_delta);
         size_t idx = 0;
         while (idx < delta_count) {
             uint32_t cid = deltas[idx].id;
             int64_t sum = 0;
             while (idx < delta_count && deltas[idx].id == cid) {
                 sum += deltas[idx].delta;
                 idx++;
             }
             if (cid >= total_nodes || sum == 0) {
                 continue;
             }
             int64_t new_value = (int64_t)comm_tot_degree[cid] + sum;
             if (new_value < 0) {
                 new_value = 0;
             }
             comm_tot_degree[cid] = (uint32_t)new_value;
             if (!u32_array_push(out_tot_ids, cid) ||
                 !u32_array_push(out_tot_vals, (uint32_t)new_value)) {
                 free(deltas);
                 ok = false;
                 goto out;
             }
         }
     }
 
     free(deltas);
 out:
     g_profile.merge_changes += profile_time_now() - start;
     return ok;
 }
 
 static bool push_unique_id(u32_array_t *arr, uint32_t id, uint8_t *marks, size_t marks_len) {
     if (marks_len == 0 || marks == NULL) {
         return u32_array_push(arr, id);
     }
     if (id >= marks_len) {
         return false;
     }
     if (marks[id]) {
         return true;
     }
     if (!u32_array_push(arr, id)) {
         return false;
     }
     marks[id] = 1;
     return true;
 }
 // 读取membership文件，返回每个顶点所属的社区
 bool load_membership_file(const char *path,
                           uint32_t expected_vertices,
                           uint32_t **out_membership,
                           uint32_t *out_num_communities) {
     FILE *fp = fopen(path, "r");
     if (!fp) {
         perror("Failed to open membership file");
         return false;
     }
     uint32_t *membership = malloc(expected_vertices * sizeof(uint32_t));
     if (!membership) {
         fprintf(stderr, "Failed to allocate membership array\n");
         fclose(fp);
         return false;
     }
     for (uint32_t i = 0; i < expected_vertices; ++i) {
         membership[i] = UINT32_MAX;
     }
 
     uint64_t assigned = 0;
     uint32_t max_comm = 0;
     uint32_t next_vertex = 0;
 
     char line[256];
     while (fgets(line, sizeof(line), fp)) {
         if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
             continue;
         }
         uint32_t idx = UINT32_MAX;
         uint32_t comm = UINT32_MAX;
         int consumed = sscanf(line, "%u %u", &idx, &comm);
         if (consumed == 1) {
             comm = idx;
             idx = next_vertex;
         } else if (consumed != 2) {
             fprintf(stderr, "Malformed membership line: %s\n", line);
             free(membership);
             fclose(fp);
             return false;
         }
         if (idx >= expected_vertices) {
             fprintf(stderr, "Membership index %u out of range (N=%u)\n", idx, expected_vertices);
             free(membership);
             fclose(fp);
             return false;
         }
         if (membership[idx] != UINT32_MAX) {
             fprintf(stderr, "Duplicate membership entry for vertex %u\n", idx);
             free(membership);
             fclose(fp);
             return false;
         }
         membership[idx] = comm;
         if (comm > max_comm) {
             max_comm = comm;
         }
         assigned++;
         if (idx == next_vertex) {
             next_vertex++;
         }
     }
     fclose(fp);
 
     if (assigned == 0) {
         fprintf(stderr, "Membership file appears to be empty\n");
         free(membership);
         return false;
     }
     for (uint32_t i = 0; i < expected_vertices; ++i) {
         if (membership[i] == UINT32_MAX) {
             fprintf(stderr, "Membership missing entry for vertex %u\n", i);
             free(membership);
             return false;
         }
     }
 
     *out_membership = membership;
     if (out_num_communities) {
         *out_num_communities = max_comm + 1;
     }
     return true;
 }
 
 static int compare_block_desc(const void *a, const void *b) {
     const community_block_t *ba = (const community_block_t *)a;
     const community_block_t *bb = (const community_block_t *)b;
     if (bb->size != ba->size) {
         return (bb->size > ba->size) ? 1 : -1;
     }
     if (bb->edge_load != ba->edge_load) {
         return (bb->edge_load > ba->edge_load) ? 1 : -1;
     }
     return (ba->id < bb->id) ? -1 : (ba->id > bb->id);
 }
 
 community_span_t *reorder_graph_by_membership(graph_t *graph,
                                               uint32_t **node_degree_ptr,
                                               uint32_t **node_comm_ptr,
                                               const uint32_t *membership,
                                               uint32_t num_communities) {
     uint32_t n = graph->n_nodes;
     if (n == 0 || !membership) {
         return NULL;
     }
     uint32_t *degree = *node_degree_ptr;
     if (!degree) {
         return NULL;
     }
 
     uint32_t *counts = calloc(num_communities, sizeof(uint32_t));
     uint64_t *edge_sums = calloc(num_communities, sizeof(uint64_t));
     community_block_t *blocks = calloc(num_communities, sizeof(community_block_t));
     if (!counts || !edge_sums || !blocks) {
         free(counts);
         free(edge_sums);
         free(blocks);
         return NULL;
     }
 
     for (uint32_t cid = 0; cid < num_communities; ++cid) {
         blocks[cid].id = cid;
     }
     for (uint32_t v = 0; v < n; ++v) {
         uint32_t cid = membership[v];
         if (cid >= num_communities) {
             cid = num_communities - 1;
         }
         counts[cid]++;
         edge_sums[cid] += degree[v];
     }
     for (uint32_t cid = 0; cid < num_communities; ++cid) {
         blocks[cid].size = counts[cid];
         blocks[cid].edge_load = edge_sums[cid];
     }
 
     qsort(blocks, num_communities, sizeof(community_block_t), compare_block_desc);
 
     uint32_t *rank = malloc(num_communities * sizeof(uint32_t));
     uint32_t cumulative = 0;
     for (uint32_t idx = 0; idx < num_communities; ++idx) {
         blocks[idx].offset = cumulative;
         rank[blocks[idx].id] = idx;
         cumulative += blocks[idx].size;
     }
     if (cumulative != n) {
         fprintf(stderr, "Membership counts (%u) mismatch node count (%u)\n", cumulative, n);
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         return NULL;
     }
 
     uint32_t *next_slot = calloc(num_communities, sizeof(uint32_t));
     uint32_t *new_to_old = malloc(n * sizeof(uint32_t));
     uint32_t *old_to_new = malloc(n * sizeof(uint32_t));
     if (!next_slot || !new_to_old || !old_to_new) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         return NULL;
     }
     for (uint32_t old_v = 0; old_v < n; ++old_v) {
         uint32_t cid = membership[old_v];
         if (cid >= num_communities) {
             cid = num_communities - 1;
         }
         uint32_t block_idx = rank[cid];
         uint32_t pos = blocks[block_idx].offset + next_slot[cid]++;
         new_to_old[pos] = old_v;
         old_to_new[old_v] = pos;
     }
 
     uint32_t *new_row_ptr = malloc((n + 1) * sizeof(uint32_t));
     uint32_t *new_col_idx = malloc(graph->n_edges * sizeof(uint32_t));
     uint32_t *cursor = malloc(n * sizeof(uint32_t));
     if (!new_row_ptr || !new_col_idx || !cursor) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         free(new_row_ptr);
         free(new_col_idx);
         free(cursor);
         return NULL;
     }
     new_row_ptr[0] = 0;
     for (uint32_t new_v = 0; new_v < n; ++new_v) {
         uint32_t old_v = new_to_old[new_v];
         uint32_t deg = graph->row_ptr[old_v + 1] - graph->row_ptr[old_v];
         new_row_ptr[new_v + 1] = new_row_ptr[new_v] + deg;
         cursor[new_v] = new_row_ptr[new_v];
     }
     for (uint32_t new_v = 0; new_v < n; ++new_v) {
         uint32_t old_v = new_to_old[new_v];
         for (uint32_t ei = graph->row_ptr[old_v]; ei < graph->row_ptr[old_v + 1]; ++ei) {
             uint32_t old_nb = graph->col_idx[ei];
             uint32_t new_nb = old_to_new[old_nb];
             uint32_t pos = cursor[new_v]++;
             new_col_idx[pos] = new_nb;
         }
     }
 
     uint32_t *new_degree = malloc(n * sizeof(uint32_t));
     if (!new_degree) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         free(new_row_ptr);
         free(new_col_idx);
         free(cursor);
         return NULL;
     }
     for (uint32_t new_v = 0; new_v < n; ++new_v) {
         new_degree[new_v] = degree[new_to_old[new_v]];
     }
 
     uint32_t *new_comm = malloc(n * sizeof(uint32_t));
     if (!new_comm) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         free(new_row_ptr);
         free(new_col_idx);
         free(cursor);
         free(new_degree);
         return NULL;
     }
     for (uint32_t new_v = 0; new_v < n; ++new_v) {
         uint32_t old_v = new_to_old[new_v];
         uint32_t cid = membership[old_v];
         if (cid >= num_communities) {
             cid = num_communities - 1;
         }
         new_comm[new_v] = rank[cid];
     }
 
     uint32_t *old_mapping = graph->new_to_old;
     uint32_t *new_mapping = malloc(n * sizeof(uint32_t));
     if (!new_mapping) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         free(new_row_ptr);
         free(new_col_idx);
         free(cursor);
         free(new_degree);
         free(new_comm);
         return NULL;
     }
     for (uint32_t new_v = 0; new_v < n; ++new_v) {
         uint32_t old_v = new_to_old[new_v];
         uint32_t original = old_mapping ? old_mapping[old_v] : old_v;
         new_mapping[new_v] = original;
     }
 
     uint32_t *old_row_ptr = graph->row_ptr;
     uint32_t *old_col_idx = graph->col_idx;
     graph->row_ptr = new_row_ptr;
     graph->col_idx = new_col_idx;
     free(old_row_ptr);
     free(old_col_idx);
     graph->new_to_old = new_mapping;
 
     free(*node_degree_ptr);
     *node_degree_ptr = new_degree;
     free(*node_comm_ptr);
     *node_comm_ptr = new_comm;
 
     community_span_t *spans = calloc(num_communities, sizeof(community_span_t));
     if (!spans) {
         free(counts);
         free(edge_sums);
         free(blocks);
         free(rank);
         free(next_slot);
         free(new_to_old);
         free(old_to_new);
         free(cursor);
         return NULL;
     }
     for (uint32_t idx = 0; idx < num_communities; ++idx) {
         spans[idx].new_id = idx;
         spans[idx].original_id = blocks[idx].id;
         spans[idx].start = blocks[idx].offset;
         spans[idx].length = blocks[idx].size;
         spans[idx].degree_sum = blocks[idx].edge_load;
     }
 
     free(counts);
     free(edge_sums);
     free(blocks);
     free(rank);
     free(next_slot);
     free(new_to_old);
     free(old_to_new);
     free(cursor);
     return spans;
 }
 
 bool assign_communities_to_dpus(const community_span_t *spans,
                                 uint32_t num_communities,
                                 uint32_t num_vertices,
                                 uint32_t num_dpus,
                                 uint32_t **out_vertex_owner,
                                 uint32_t **out_community_owner,
                                 uint32_t **out_dpu_starts,
                                 uint32_t **out_dpu_counts) {
     if (!spans || num_dpus == 0) {
         return false;
     }
     uint32_t *vertex_owner = malloc(num_vertices * sizeof(uint32_t));
     uint32_t *community_owner = malloc(num_communities * sizeof(uint32_t));
     uint32_t *dpu_starts = malloc(num_dpus * sizeof(uint32_t));
     uint32_t *dpu_counts = calloc(num_dpus, sizeof(uint32_t));
     uint64_t *dpu_edge_load = calloc(num_dpus, sizeof(uint64_t));
     if (!vertex_owner || !community_owner || !dpu_starts || !dpu_counts || !dpu_edge_load) {
         free(vertex_owner);
         free(community_owner);
         free(dpu_starts);
         free(dpu_counts);
         free(dpu_edge_load);
         return false;
     }
     for (uint32_t i = 0; i < num_vertices; ++i) {
         vertex_owner[i] = UINT32_MAX;
     }
     for (uint32_t d = 0; d < num_dpus; ++d) {
         dpu_starts[d] = UINT32_MAX;
     }
 
     uint32_t current_dpu = 0;
     for (uint32_t idx = 0; idx < num_communities; ++idx) {
         uint32_t size = spans[idx].length;
         if (size == 0) {
             community_owner[idx] = current_dpu;
             continue;
         }
         uint64_t edge_load = spans[idx].degree_sum;
 
         while (current_dpu < num_dpus &&
                dpu_counts[current_dpu] > 0 &&
                (dpu_counts[current_dpu] + size > MAX_NODES_PER_DPU ||
                 dpu_edge_load[current_dpu] + edge_load > MAX_EDGES_PER_DPU)) {
             current_dpu++;
         }
         if (current_dpu >= num_dpus) {
             fprintf(stderr, "Insufficient DPUs to place community %u (size=%u)\n",
                     spans[idx].original_id, size);
             free(vertex_owner);
             free(community_owner);
             free(dpu_starts);
             free(dpu_counts);
             free(dpu_edge_load);
             return false;
         }
         if (dpu_counts[current_dpu] == 0) {
             if (size > MAX_NODES_PER_DPU || edge_load > MAX_EDGES_PER_DPU) {
                 fprintf(stderr,
                         "Community %u (size=%u, edges=%" PRIu64 ") exceeds DPU capacity\n",
                         spans[idx].original_id, size, edge_load);
                 free(vertex_owner);
                 free(community_owner);
                 free(dpu_starts);
                 free(dpu_counts);
                 free(dpu_edge_load);
                 return false;
             }
             dpu_starts[current_dpu] = spans[idx].start;
         }
 
         community_owner[idx] = current_dpu;
         dpu_counts[current_dpu] += size;
         dpu_edge_load[current_dpu] += edge_load;
         for (uint32_t off = 0; off < size; ++off) {
             uint32_t global_vertex = spans[idx].start + off;
             vertex_owner[global_vertex] = current_dpu;
         }
     }
 
     for (uint32_t d = 0; d < num_dpus; ++d) {
         if (dpu_starts[d] == UINT32_MAX) {
             dpu_starts[d] = (d > 0) ? (dpu_starts[d - 1] + dpu_counts[d - 1]) : 0;
             dpu_counts[d] = 0;
         }
     }
 
     free(dpu_edge_load);
     *out_vertex_owner = vertex_owner;
     *out_community_owner = community_owner;
     *out_dpu_starts = dpu_starts;
     *out_dpu_counts = dpu_counts;
     return true;
 }
 
 static bool prepare_incremental_worklists(const edge_update_t *updates,
                                           size_t update_count,
                                           const uint32_t *vertex_comm,
                                           const uint32_t *vertex_owner,
                                           const uint32_t *community_owner,
                                           uint32_t total_vertices,
                                           uint32_t total_communities,
                                           uint32_t num_dpus,
                                           dpu_incremental_work_t *worklists,
                                           remote_notification_t *notifications) {
     if (num_dpus == 0 || total_vertices == 0) {
         return true;
     }
     for (uint32_t d = 0; d < num_dpus; ++d) {
         dpu_incremental_work_clear(&worklists[d]);
         remote_notification_clear(&notifications[d]);
     }
     uint8_t *vertex_marks = calloc(total_vertices, sizeof(uint8_t));
     uint8_t *community_marks = calloc(total_communities, sizeof(uint8_t));
     if ((!vertex_marks && total_vertices) || (!community_marks && total_communities)) {
         free(vertex_marks);
         free(community_marks);
         return false;
     }
     bool ok = true;
     for (size_t idx = 0; idx < update_count && ok; ++idx) {
         const edge_update_t *up = &updates[idx];
         uint32_t u = up->src;
         uint32_t v = up->dst;
         if (u >= total_vertices || v >= total_vertices) {
             continue;
         }
         uint32_t comm_u = vertex_comm ? vertex_comm[u] : UINT32_MAX;
         uint32_t comm_v = vertex_comm ? vertex_comm[v] : UINT32_MAX;
         uint32_t owner_u = vertex_owner ? vertex_owner[u] : 0;
         uint32_t owner_v = vertex_owner ? vertex_owner[v] : owner_u;
         if (owner_u >= num_dpus || owner_v >= num_dpus) {
             continue;
         }
         dpu_incremental_work_t *work_u = &worklists[owner_u];
         if (!u32_array_push(&work_u->update_indices, (uint32_t)idx)) {
             ok = false;
             break;
         }
         if (!push_unique_id(&work_u->vertices, u, vertex_marks, total_vertices)) {
             ok = false;
             break;
         }
         if (owner_u == owner_v) {
             if (!push_unique_id(&work_u->vertices, v, vertex_marks, total_vertices)) {
                 ok = false;
                 break;
             }
         } else {
             dpu_incremental_work_t *work_v = &worklists[owner_v];
             if (!u32_array_push(&work_v->update_indices, (uint32_t)idx)) {
                 ok = false;
                 break;
             }
             if (!push_unique_id(&work_v->vertices, v, vertex_marks, total_vertices)) {
                 ok = false;
                 break;
             }
         }
         bool same_comm = (comm_u == comm_v) && (comm_u != UINT32_MAX);
         if (up->type == EDGE_DELETION) {
             if (same_comm) {
                 uint32_t comm_owner = community_owner ? community_owner[comm_u] : owner_u;
                 if (comm_owner < num_dpus) {
                     if (!push_unique_id(&worklists[comm_owner].communities, comm_u,
                                         community_marks, total_communities)) {
                         ok = false;
                         break;
                     }
                 }
             } else {
                 if (comm_u != UINT32_MAX) {
                     uint32_t comm_owner = community_owner ? community_owner[comm_u] : owner_u;
                     if (comm_owner < num_dpus) {
                         if (!push_unique_id(&worklists[comm_owner].communities, comm_u,
                                             community_marks, total_communities)) {
                             ok = false;
                             break;
                         }
                     }
                 }
                 if (comm_v != UINT32_MAX) {
                     uint32_t comm_owner = community_owner ? community_owner[comm_v] : owner_v;
                     if (comm_owner < num_dpus) {
                         if (!push_unique_id(&worklists[comm_owner].communities, comm_v,
                                             community_marks, total_communities)) {
                             ok = false;
                             break;
                         }
                     }
                 }
             }
             continue;
         }
         if (same_comm) {
             uint32_t comm_owner = community_owner ? community_owner[comm_u] : owner_u;
             if (comm_owner < num_dpus) {
                 if (!push_unique_id(&worklists[comm_owner].communities, comm_u,
                                     community_marks, total_communities)) {
                     ok = false;
                     break;
                 }
             }
             continue;
         }
         if (!u32_array_push(&work_u->candidate_comms, comm_v)) {
             ok = false;
             break;
         }
         if (comm_u != UINT32_MAX) {
             uint32_t comm_owner_u = community_owner ? community_owner[comm_u] : owner_u;
             if (comm_owner_u < num_dpus) {
                 if (!push_unique_id(&worklists[comm_owner_u].communities, comm_u,
                                     community_marks, total_communities)) {
                     ok = false;
                     break;
                 }
             }
         }
         uint32_t comm_owner_v = community_owner ? community_owner[comm_v] : owner_v;
         if (comm_owner_v < num_dpus) {
             if (!push_unique_id(&worklists[comm_owner_v].communities, comm_v,
                                 community_marks, total_communities)) {
                 ok = false;
                 break;
             }
         }
         if (owner_u != owner_v) {
             if (!u32_array_push(&notifications[owner_u].comm_ids, comm_v) ||
                 !u32_array_push(&notifications[owner_u].target_dpus, comm_owner_v)) {
                 ok = false;
                 break;
             }
         }
     }
     /* Sort per-DPU vertex lists by CSR row offset for better locality */
     if (g_row_ptr_for_sort) {
         for (uint32_t d = 0; d < num_dpus; ++d) {
             u32_array_t *verts = &worklists[d].vertices;
             if (verts->size > 1) {
                 qsort(verts->data, verts->size, sizeof(uint32_t), compare_by_rowptr);
             }
         }
     }
     free(vertex_marks);
     free(community_marks);
     return ok;
 }
 
 /**
  * Prepare per-DPU worklists for an incremental batch using the ownership maps
  * derived from the offline Leiden partitioning stage. Returns false if the
  * batch cannot be processed (e.g. mapping unavailable).
  */
 bool process_incremental_batch(const edge_update_t *updates,
                                size_t update_count,
                                const uint32_t *current_membership,
                                uint32_t total_vertices,
                                incremental_context_t *ctx) {
     if (!ctx || ctx->num_dpus == 0) {
         return false;
     }
     if (!updates || update_count == 0) {
         incremental_context_clear(ctx);
         return true;
     }
     if (!current_membership || !g_vertex_owner || !g_community_owner || g_num_communities == 0) {
         return false;
     }
     incremental_context_clear(ctx);
     return prepare_incremental_worklists(updates,
                                          update_count,
                                          current_membership,
                                          g_vertex_owner,
                                          g_community_owner,
                                          total_vertices,
                                          g_num_communities,
                                          ctx->num_dpus,
                                          ctx->worklists,
                                         ctx->notifications);
 }
 
// fraction_percent: 0-10000 表示 0%-100.00%
size_t synthesize_random_updates(edge_update_t *buffer,
                                 size_t capacity,
                                 uint32_t num_vertices,
                                 uint64_t epoch,
                                 uint32_t fraction_percent) {
    if (!buffer || capacity == 0 || num_vertices < 2) {
        return 0;
    }

    if (fraction_percent == 0) {
        size_t count = capacity < 8 ? capacity : 8;
        size_t actual = count > 0 ? count : 1;
        uint64_t seed = 0x9e3779b97f4a7c15ULL ^ epoch ^ (uint64_t)num_vertices;
        uint64_t state = seed ? seed : 0xdeadbeefULL;
        for (size_t i=0; i<actual; ++i) {
            uint32_t u = (uint32_t)(prng_step(&state) % num_vertices);
            uint32_t v = (uint32_t)(prng_step(&state) % num_vertices);
            if (u == v) v = (v + 1) % num_vertices;
            if (u > v) { uint32_t tmp = u; u = v; v = tmp; }
            uint32_t w = 1 + (uint32_t)(prng_step(&state) % 8);
            edge_update_type_t type = (prng_step(&state) & 1ULL) ? EDGE_INSERTION : EDGE_DELETION;
            buffer[i].type = type;
            buffer[i].src = u;
            buffer[i].dst = v;
            buffer[i].weight = w;
        }
        return actual;
    }

    uint32_t frac_pct = fraction_percent;
    if (frac_pct > 10000) frac_pct = 10000;

    size_t target = ((uint64_t)num_vertices * frac_pct) / 10000;
    if (target == 0) target = 1;
    if (target > capacity) target = capacity;
    if (target > num_vertices) target = num_vertices;
 
     uint64_t seed = 0x9e3779b97f4a7c15ULL ^ epoch ^ ((uint64_t)num_vertices << 7);
     uint64_t state = seed ? seed : 0x12345678ULL;
     uint32_t start = (uint32_t)(prng_step(&state) % num_vertices);
 
     for (size_t i=0; i<target; ++i) {
         uint32_t u = (start + (uint32_t)i) % num_vertices;
         uint32_t shift = (uint32_t)(prng_step(&state) % (num_vertices - 1)) + 1;
         uint32_t v = (u + shift) % num_vertices;
         if (u > v) { uint32_t tmp = u; u = v; v = tmp; }
         buffer[i].type = EDGE_INSERTION;
         buffer[i].src = u;
         buffer[i].dst = v;
         buffer[i].weight = 1 + (uint32_t)(prng_step(&state) % 8);
     }
     return target;
 }
 
 bool broadcast_incremental_worklists_to_dpus(struct dpu_set_t dpu_set,
                                              incremental_context_t *ctx) {
     if (!ctx || ctx->num_dpus == 0) {
         return true;
     }
     uint32_t idx = 0;
     struct dpu_set_t dpu;
     DPU_FOREACH(dpu_set, dpu) {
         if (idx >= ctx->num_dpus) {
             break;
         }
         dpu_incremental_work_t *work = &ctx->worklists[idx];
         uint32_t vertex_count = (uint32_t)work->vertices.size;
         if (vertex_count > MAX_INCREMENTAL_ITEMS) {
             fprintf(stderr, "Incremental payload exceeds: %u MAX_INCREMENTAL_ITEMS(%u) for DPU %u\n", vertex_count, MAX_INCREMENTAL_ITEMS, idx);
             return false;
         }
         if (vertex_count > 1 && g_row_ptr_for_sort) {
             qsort(work->vertices.data, vertex_count, sizeof(uint32_t), compare_by_rowptr);
         }
         if (vertex_count > 0) {
             if (dpu_copy_padded(dpu, SYMBOL_INC_VERTEX_IDS, 0,
                                 work->vertices.data, vertex_count * sizeof(uint32_t)) != DPU_OK) {
                 fprintf(stderr, "Failed to copy incremental vertices to DPU %u\n", idx);
                 return false;
             }
         }
         if (dpu_copy_to(dpu, SYMBOL_INC_VERTEX_COUNT, 0,
                         &vertex_count, sizeof(vertex_count)) != DPU_OK) {
             fprintf(stderr, "Failed to copy incremental counts to DPU %u\n", idx);
             return false;
         }
         idx++;
     }
     return true;
 }
 
 bool load_graph(const char *filename, graph_t *graph) {
     memset(graph, 0, sizeof(*graph));
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         perror("Unable to open graph file");
         return false;
     }
 
     double read_start = profile_time_now();
     uint32_t hint_nodes = 0, hint_edges = 0;
     if (fscanf(fp, "%u %u", &hint_nodes, &hint_edges) != 2) {
         fprintf(stderr, "Graph file format error (expected N M on first line)\n");
         fclose(fp);
         g_profile.load_read_edges += profile_time_now() - read_start;
         return false;
     }
 
     uint32_t capacity_edges = hint_edges ? hint_edges : 1024;
     uint32_t *src = malloc(capacity_edges * sizeof(uint32_t));
     uint32_t *dst = malloc(capacity_edges * sizeof(uint32_t));
     if (!src || !dst) {
         fprintf(stderr, "Memory allocation failed while staging edges\n");
         fclose(fp);
         free(src);
         free(dst);
         return false;
     }
 
     uint32_t edge_index = 0;
     uint32_t max_node_id = (hint_nodes > 0) ? (hint_nodes - 1) : 0;
     while (true) {
         uint32_t u, v;
         int read = fscanf(fp, "%u %u", &u, &v);
         if (read == EOF) {
             break;
         }
         if (read != 2) {
             fprintf(stderr, "Graph file format error around edge %u\n", edge_index);
             free(src);
             free(dst);
             fclose(fp);
             g_profile.load_read_edges += profile_time_now() - read_start;
             return false;
         }
         if (edge_index == capacity_edges) {
             uint32_t new_capacity = capacity_edges * 2;
             uint32_t *new_src = realloc(src, new_capacity * sizeof(uint32_t));
             uint32_t *new_dst = realloc(dst, new_capacity * sizeof(uint32_t));
             if (!new_src || !new_dst) {
                 fprintf(stderr, "Edge buffer reallocation failed\n");
                 free(new_src ? new_src : src);
                 free(new_dst ? new_dst : dst);
                 fclose(fp);
                 g_profile.load_read_edges += profile_time_now() - read_start;
                 return false;
             }
             src = new_src;
             dst = new_dst;
             capacity_edges = new_capacity;
         }
         src[edge_index] = u;
         dst[edge_index] = v;
         if (u > max_node_id) {
             max_node_id = u;
         }
         if (v > max_node_id) {
             max_node_id = v;
         }
         edge_index++;
     }
     fclose(fp);
 
     if (edge_index == 0) {
         fprintf(stderr, "The graph contains no edges.\n");
         free(src);
         free(dst);
         g_profile.load_read_edges += profile_time_now() - read_start;
         return false;
     }
 
     double read_end = profile_time_now();
     g_profile.load_read_edges += read_end - read_start;
     double build_start = read_end;
 
     size_t pair_count = (size_t)edge_index;
     uint32_t *all_ids = malloc(pair_count * 2U * sizeof(uint32_t));
     if (!all_ids) {
         fprintf(stderr, "Memory allocation failed while staging node IDs\n");
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
     for (size_t i = 0; i < pair_count; ++i) {
         all_ids[2 * i] = src[i];
         all_ids[2 * i + 1] = dst[i];
     }
     qsort(all_ids, pair_count * 2U, sizeof(uint32_t), compare_u32);
 
     uint32_t unique_count = 0;
     for (size_t i = 0; i < pair_count * 2U; ++i) {
         if (i == 0 || all_ids[i] != all_ids[i - 1]) {
             all_ids[unique_count++] = all_ids[i];
         }
     }
     if (unique_count == 0) {
         fprintf(stderr, "The graph contains no valid nodes.\n");
         free(all_ids);
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
 
     uint32_t *new_to_old = malloc(unique_count * sizeof(uint32_t));
     if (!new_to_old) {
         fprintf(stderr, "Failed to allocate ID remap array.\n");
         free(all_ids);
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
     memcpy(new_to_old, all_ids, unique_count * sizeof(uint32_t));
 
     uint32_t max_old_id = new_to_old[unique_count - 1];
     uint32_t *old_to_new = malloc(((size_t)max_old_id + 1U) * sizeof(uint32_t));
     if (!old_to_new) {
         fprintf(stderr, "Failed to allocate reverse ID mapping.\n");
         free(new_to_old);
         free(all_ids);
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
     for (uint32_t i = 0; i <= max_old_id; ++i) {
         old_to_new[i] = UINT32_MAX;
     }
     for (uint32_t i = 0; i < unique_count; ++i) {
         old_to_new[new_to_old[i]] = i;
     }
 
     for (uint32_t i = 0; i < edge_index; ++i) {
         uint32_t u = src[i];
         uint32_t v = dst[i];
         uint32_t nu = (u <= max_old_id) ? old_to_new[u] : UINT32_MAX;
         uint32_t nv = (v <= max_old_id) ? old_to_new[v] : UINT32_MAX;
         if (nu == UINT32_MAX || nv == UINT32_MAX) {
             fprintf(stderr, "Edge references unknown node (%u,%u).\n", u, v);
             free(old_to_new);
             free(new_to_old);
             free(all_ids);
             free(src);
             free(dst);
             g_profile.load_build_csr += profile_time_now() - build_start;
             return false;
         }
         src[i] = nu;
         dst[i] = nv;
     }
 
     free(old_to_new);
     free(all_ids);
 
     graph->row_ptr = calloc(unique_count + 1U, sizeof(uint32_t));
     graph->col_idx = malloc(edge_index * sizeof(uint32_t));
     if (!graph->row_ptr || !graph->col_idx) {
         fprintf(stderr, "Memory allocation failed while building CSR graph\n");
         free_graph(graph);
         free(new_to_old);
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
 
     for (uint32_t e = 0; e < edge_index; ++e) {
         uint32_t u = src[e];
         graph->row_ptr[u + 1]++;
     }
     for (uint32_t i = 0; i < unique_count; ++i) {
         graph->row_ptr[i + 1] += graph->row_ptr[i];
     }
 
     uint32_t *cursor = malloc(unique_count * sizeof(uint32_t));
     if (!cursor) {
         fprintf(stderr, "Failed to allocate cursor buffer\n");
         free_graph(graph);
         free(new_to_old);
         free(src);
         free(dst);
         g_profile.load_build_csr += profile_time_now() - build_start;
         return false;
     }
     memcpy(cursor, graph->row_ptr, unique_count * sizeof(uint32_t));
 
     for (uint32_t e = 0; e < edge_index; ++e) {
         uint32_t u = src[e];
         uint32_t v = dst[e];
         uint32_t pos = cursor[u]++;
         graph->col_idx[pos] = v;
     }
 
     free(cursor);
     free(src);
     free(dst);
 
     graph->n_nodes = unique_count;
     graph->n_edges = edge_index;
     graph->new_to_old = new_to_old;
     g_profile.load_build_csr += profile_time_now() - build_start;
     return true;
 }
 
 uint32_t *compute_degrees(const graph_t *graph) {
     uint32_t *degrees = malloc(graph->n_nodes * sizeof(uint32_t));
     if (!degrees) {
         return NULL;
     }
     for (uint32_t i = 0; i < graph->n_nodes; ++i) {
         degrees[i] = graph->row_ptr[i + 1] - graph->row_ptr[i];
     }
     return degrees;
 }
 
 void free_partitions(dpu_partition_t *partitions, uint32_t num_dpus) {
     if (!partitions) {
         return;
     }
     for (uint32_t i = 0; i < num_dpus; ++i) {
         free(partitions[i].node_degree_local);
     }
     free(partitions);
 }
 
 dpu_partition_t *setup_partitions(const graph_t *graph,
                                   const uint32_t *node_degree,
                                   const uint32_t *node_comm,
                                   struct dpu_set_t dpu_set,
                                   uint32_t num_dpus,
                                   const uint32_t *vertex_starts,
                                   const uint32_t *vertex_counts) {
     dpu_partition_t *partitions = calloc(num_dpus, sizeof(*partitions));
     if (!partitions) {
         return NULL;
     }
 
     uint32_t dpu_id = 0;
     struct dpu_set_t dpu;
     DPU_FOREACH(dpu_set, dpu) {
         dpu_partition_t *part = &partitions[dpu_id];
         uint32_t start_node = 0;
         uint32_t local_nodes = 0;
         if (vertex_starts && vertex_counts) {
             start_node = (dpu_id < num_dpus) ? vertex_starts[dpu_id] : 0;
             local_nodes = (dpu_id < num_dpus) ? vertex_counts[dpu_id] : 0;
         } else {
             uint32_t nodes_per_dpu = (graph->n_nodes + num_dpus - 1) / num_dpus;
             start_node = dpu_id * nodes_per_dpu;
             uint32_t end_node = start_node + nodes_per_dpu;
             if (end_node > graph->n_nodes) {
                 end_node = graph->n_nodes;
             }
             local_nodes = (start_node < graph->n_nodes) ? (end_node - start_node) : 0;
         }
         if (start_node > graph->n_nodes) {
             start_node = graph->n_nodes;
             local_nodes = 0;
         }
         if (start_node + local_nodes > graph->n_nodes) {
             local_nodes = graph->n_nodes - start_node;
         }
         uint32_t start_edge = (start_node < graph->n_nodes) ? graph->row_ptr[start_node] : 0;
         uint32_t edge_count = 0;
         if (local_nodes > 0) {
             edge_count = graph->row_ptr[start_node + local_nodes] - start_edge;
         }
 
         if (local_nodes > MAX_NODES_PER_DPU) {
             fprintf(stderr,
                     "Partition %u exceeds MAX_NODES_PER_DPU (%u > %u); recompile with larger limit.\n",
                     dpu_id, local_nodes, MAX_NODES_PER_DPU);
             free_partitions(partitions, num_dpus);
             return NULL;
         }
         if (edge_count > MAX_EDGES_PER_DPU) {
             fprintf(stderr,
                     "Partition %u exceeds MAX_EDGES_PER_DPU (%u > %u); recompile with larger limit.\n",
                     dpu_id, edge_count, MAX_EDGES_PER_DPU);
             free_partitions(partitions, num_dpus);
             return NULL;
         }
 
         part->start_node = start_node;
         part->local_nodes = local_nodes;
         part->start_edge = start_edge;
         part->edge_count = edge_count;
 
         if (edge_count > 0 &&
             dpu_copy_padded(dpu, "neighbors", 0, graph->col_idx + start_edge,
                             edge_count * sizeof(uint32_t)) != DPU_OK) {
             fprintf(stderr, "Failed to copy neighbors to DPU %u\n", dpu_id);
             free_partitions(partitions, num_dpus);
             return NULL;
         }
 
         if (local_nodes > 0) {
             part->node_degree_local = malloc(local_nodes * sizeof(uint32_t));
             if (!part->node_degree_local) {
                 free_partitions(partitions, num_dpus);
                 return NULL;
             }
             for (uint32_t i = 0; i < local_nodes; ++i) {
                 part->node_degree_local[i] = node_degree[start_node + i];
             }
 
             uint32_t *local_row_ptr = malloc((local_nodes + 1) * sizeof(uint32_t));
             if (!local_row_ptr) {
                 free_partitions(partitions, num_dpus);
                 return NULL;
             }
             for (uint32_t i = 0; i <= local_nodes; ++i) {
                 uint32_t global_node = start_node + i;
                 if (global_node > graph->n_nodes) {
                     global_node = graph->n_nodes;
                 }
                 uint32_t global_edge_index = graph->row_ptr[global_node];
                 local_row_ptr[i] = (global_edge_index > start_edge) ? (global_edge_index - start_edge) : 0;
             }
 
             if (dpu_copy_to(dpu, "base_id", 0, &start_node, sizeof(start_node)) != DPU_OK ||
                 dpu_copy_to(dpu, "node_count", 0, &local_nodes, sizeof(local_nodes)) != DPU_OK ||
                 dpu_copy_padded(dpu, "node_row_ptr", 0, local_row_ptr,
                                 (local_nodes + 1) * sizeof(uint32_t)) != DPU_OK ||
                 dpu_copy_padded(dpu, "node_degree_local", 0, part->node_degree_local,
                                 local_nodes * sizeof(uint32_t)) != DPU_OK ||
                 dpu_copy_padded(dpu, "node_comm_local", 0, node_comm + start_node,
                                 local_nodes * sizeof(uint32_t)) != DPU_OK) {
                 fprintf(stderr, "Failed to initialize DPU %u metadata\n", dpu_id);
                 free(local_row_ptr);
                 free_partitions(partitions, num_dpus);
                 return NULL;
             }
             free(local_row_ptr);
         }
 
         dpu_id++;
     }
 
     return partitions;
 }
 
 bool compute_comm_totals(const graph_t *graph,
                          const uint32_t *node_degree,
                          const uint32_t *node_comm,
                          uint32_t *comm_tot_degree) {
     double start = profile_time_now();
     if (!comm_tot_degree) {
         g_profile.compute_totals += profile_time_now() - start;
         return false;
     }
     memset(comm_tot_degree, 0, graph->n_nodes * sizeof(uint32_t));
     for (uint32_t node = 0; node < graph->n_nodes; ++node) {
         uint32_t cid = node_comm[node];
         if (cid >= graph->n_nodes) {
             g_profile.compute_totals += profile_time_now() - start;
             return false;
         }
         uint64_t accum = (uint64_t)comm_tot_degree[cid] + node_degree[node];
         comm_tot_degree[cid] = (accum > UINT32_MAX) ? UINT32_MAX : (uint32_t)accum;
     }
     g_profile.compute_totals += profile_time_now() - start;
     return true;
 }
 
 
 bool broadcast_phase(struct dpu_set_t dpu_set, uint32_t phase, uint32_t two_m) {
     double start = profile_time_now();
     struct dpu_set_t dpu;
     DPU_FOREACH(dpu_set, dpu) {
         if (dpu_copy_to(dpu, "phase", 0, &phase, sizeof(phase)) != DPU_OK ||
             dpu_copy_to(dpu, "two_m", 0, &two_m, sizeof(two_m)) != DPU_OK) {
             fprintf(stderr, "Failed to broadcast control parameters\n");
             g_profile.host_to_dpu_params += profile_time_now() - start;
             return false;
         }
     }
     g_profile.host_to_dpu_params += profile_time_now() - start;
     return true;
 }
 
 uint64_t run_phase(struct dpu_set_t dpu_set,
                    u32_array_t *changed_nodes,
                    u32_array_t *changed_comms,
                    u32_array_t *changed_old_comms,
                    u32_array_t *changed_degrees,
                    double *dpu_time_accum) {
     u32_array_clear(changed_nodes);
     u32_array_clear(changed_comms);
     u32_array_clear(changed_old_comms);
     u32_array_clear(changed_degrees);
     struct timespec t0, t1;
     if (dpu_time_accum) {
         clock_gettime(CLOCK_MONOTONIC, &t0);
     }
     double launch_start = profile_time_now();
     if (dpu_launch(dpu_set, DPU_SYNCHRONOUS) != DPU_OK) {
         g_profile.dpu_launch += profile_time_now() - launch_start;
         fprintf(stderr, "dpu_launch failed\n");
         return UINT64_MAX;
     }
     g_profile.dpu_launch += profile_time_now() - launch_start;
     if (dpu_time_accum) {
         clock_gettime(CLOCK_MONOTONIC, &t1);
         *dpu_time_accum += elapsed_seconds(&t0, &t1);
     }
 
    uint64_t total_moves = 0;
    double fetch_start = profile_time_now();
    struct dpu_set_t dpu;
    uint32_t idx = 0;
    uint32_t maxidx = 0;
    uint32_t max_cycle = 0;
    uint64_t best_cycles_total = 0;
    uint64_t best_cycles_1 = 0;
    uint64_t best_cycles_2 = 0;
    uint64_t best_cycles_3 = 0;
    
    // 优化：只在需要时读取cycle计数（通过环境变量控制）
    static int report_cycles = -1;
    if (report_cycles == -1) {
        report_cycles = getenv("REPORT_DPU_CYCLES") ? 1 : 0;
    }
    
    DPU_FOREACH(dpu_set, dpu) {
        //dpu分段计时（可选）
        if (report_cycles) {
            uint64_t cycles_total = 0;
            uint64_t cycles_1 = 0;
            uint64_t cycles_2 = 0;
            uint64_t cycles_3 = 0;
            dpu_error_t rc = dpu_copy_from(dpu, "cycles_total", 0, &cycles_total, sizeof(cycles_total));
            dpu_copy_from(dpu, "cycles_1", 0, &cycles_1, sizeof(cycles_1));
            dpu_copy_from(dpu, "cycles_2", 0, &cycles_2, sizeof(cycles_2));
            dpu_copy_from(dpu, "cycles_3", 0, &cycles_3, sizeof(cycles_3));
            if (rc != DPU_OK) {
                fprintf(stderr, "Failed to copy cycles from DPU %u: %d\n", idx, (int)rc);
            } else{
                if(cycles_total > max_cycle){
                    max_cycle = cycles_total;
                    maxidx = idx;
                    best_cycles_total = cycles_total;
                    best_cycles_1 = cycles_1;
                    best_cycles_2 = cycles_2;
                    best_cycles_3 = cycles_3;
                }
            }
        }
        idx++;
 
         uint32_t local_changes = 0;
         if (dpu_copy_from(dpu, "changed_count", 0, &local_changes, sizeof(local_changes)) != DPU_OK) {
             g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
             fprintf(stderr, "Failed to read changed_count from DPU\n");
             return UINT64_MAX;
         }
         total_moves += local_changes;
         if (local_changes > 0) {
             uint32_t *dst_nodes = NULL;
             uint32_t *dst_comms = NULL;
             uint32_t *dst_old = NULL;
             uint32_t *dst_degrees = NULL;
             if (!u32_array_extend(changed_nodes, local_changes, &dst_nodes) ||
                 !u32_array_extend(changed_comms, local_changes, &dst_comms) ||
                 !u32_array_extend(changed_old_comms, local_changes, &dst_old) ||
                 !u32_array_extend(changed_degrees, local_changes, &dst_degrees)) {
                 g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
                 fprintf(stderr, "Failed to allocate space for change list\n");
                 return UINT64_MAX;
             }
             if (dpu_copy_from_padded(dpu, "changed_global_ids", 0,
                                      dst_nodes,
                                      local_changes * sizeof(uint32_t)) != DPU_OK) {
                 g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
                 fprintf(stderr, "Failed to read changed_global_ids from DPU\n");
                 return UINT64_MAX;
             }
             if (dpu_copy_from_padded(dpu, "changed_new_comm", 0,
                                      dst_comms,
                                      local_changes * sizeof(uint32_t)) != DPU_OK) {
                 g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
                 fprintf(stderr, "Failed to read changed_new_comm from DPU\n");
                 return UINT64_MAX;
             }
             if (dpu_copy_from_padded(dpu, "changed_old_comm", 0,
                                      dst_old,
                                      local_changes * sizeof(uint32_t)) != DPU_OK) {
                 g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
                 fprintf(stderr, "Failed to read changed_old_comm from DPU\n");
                 return UINT64_MAX;
             }
             if (dpu_copy_from_padded(dpu, "changed_degrees", 0,
                                      dst_degrees,
                                      local_changes * sizeof(uint32_t)) != DPU_OK) {
                 g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
                 fprintf(stderr, "Failed to read changed_degrees from DPU\n");
                 return UINT64_MAX;
             }
         }
     }
    /* Convert cycles to seconds using DPU frequency (adjust DPU_FREQ_HZ if needed) */
    if (report_cycles) {
#ifndef DPU_FREQ_HZ
#define DPU_FREQ_HZ 350000000.0 /* 350 MHz default; change to your DPU core frequency */
#endif
        double secs = (double)best_cycles_total / (double)DPU_FREQ_HZ;
        double secs1 = (double)best_cycles_1 / (double)DPU_FREQ_HZ;
        double secs2 = (double)best_cycles_2 / (double)DPU_FREQ_HZ;
        double secs3 = (double)best_cycles_3 / (double)DPU_FREQ_HZ;
        printf("DPU %u cycles_total: %" PRIu64 " (%.6f s)\n", maxidx, (uint64_t)best_cycles_total, secs);
        printf("DPU %u cycles_1: %" PRIu64 " (%.6f s)\n", maxidx, (uint64_t)best_cycles_1, secs1);
        printf("DPU %u cycles_2: %" PRIu64 " (%.6f s)\n", maxidx, (uint64_t)best_cycles_2, secs2);
        printf("DPU %u cycles_3: %" PRIu64 " (%.6f s)\n", maxidx, (uint64_t)best_cycles_3, secs3);
    }
    
     g_profile.dpu_fetch_changes += profile_time_now() - fetch_start;
     return total_moves;
 }
 
 uint64_t compute_total_degree(const graph_t *graph, const uint32_t *node_degree) {
     uint64_t total = 0;
     for (uint32_t i = 0; i < graph->n_nodes; ++i) {
         total += node_degree[i];
     }
     return total;
 }
 
 double compute_graph_modularity(const graph_t *graph,
                                 const uint32_t *node_degree,
                                 const uint32_t *node_comm) {
     double start = profile_time_now();
     if (!graph || !node_degree || !node_comm || graph->n_nodes == 0) {
         g_profile.compute_modularity += profile_time_now() - start;
         return 0.0;
     }
     uint64_t two_m = 0;
     for (uint32_t i = 0; i < graph->n_nodes; ++i) {
         two_m += node_degree[i];
     }
     if (two_m == 0) {
         g_profile.compute_modularity += profile_time_now() - start;
         return 0.0;
     }
     double inv_two_m = 1.0 / (double)two_m;
     double modularity = 0.0;
     for (uint32_t u = 0; u < graph->n_nodes; ++u) {
         double deg_u = (double)node_degree[u];
         uint32_t comm_u = node_comm[u];
         for (uint32_t ei = graph->row_ptr[u]; ei < graph->row_ptr[u + 1]; ++ei) {
             uint32_t v = graph->col_idx[ei];
             if (v >= graph->n_nodes) {
                 continue;
             }
             if (node_comm[v] != comm_u) {
                 continue;
             }
             double expected = deg_u * (double)node_degree[v] * inv_two_m;
             modularity += 1.0 - expected;
         }
     }
     modularity *= inv_two_m;
     g_profile.compute_modularity += profile_time_now() - start;
     return modularity;
 }

// 边对比较函数（用于排序去重）
typedef struct { uint32_t src, dst; } edge_pair_t;

static int cmp_edges(const void *a, const void *b) {
    const edge_pair_t *ea = (const edge_pair_t*)a;
    const edge_pair_t *eb = (const edge_pair_t*)b;
    if (ea->src != eb->src) return (ea->src < eb->src) ? -1 : 1;
    if (ea->dst != eb->dst) return (ea->dst < eb->dst) ? -1 : 1;
    return 0;
}
 
int aggregate_graph(graph_t *graph,
                    uint32_t **node_degree_ptr,
                    uint32_t **node_comm_ptr,
                    uint32_t *orig_to_current,
                    uint32_t orig_nodes) {
    uint32_t old_n = graph->n_nodes;
    uint32_t *node_comm = *node_comm_ptr;
    
    // 优化：使用memset代替循环初始化
    uint32_t *comm_map = malloc(old_n * sizeof(uint32_t));
    uint32_t *node_new_comm = malloc(old_n * sizeof(uint32_t));
    if (!comm_map || !node_new_comm) {
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    memset(comm_map, 0xFF, old_n * sizeof(uint32_t));  // 0xFFFFFFFF = UINT32_MAX

    uint32_t new_n = 0;
    for (uint32_t i = 0; i < old_n; ++i) {
        uint32_t cid = node_comm[i];
        if (cid >= old_n) {
            cid = old_n - 1;
        }
        if (comm_map[cid] == UINT32_MAX) {
            comm_map[cid] = new_n++;
        }
        node_new_comm[i] = comm_map[cid];
    }
     if (new_n == old_n) {
         free(comm_map);
         free(node_new_comm);
         return 0;
     }
 
    // 优化：并行更新orig_to_current
    #pragma omp parallel for schedule(static, 65536)
    for (uint32_t i = 0; i < orig_nodes; ++i) {
        uint32_t curr = orig_to_current[i];
        curr = (curr >= old_n) ? (old_n - 1) : curr;
        orig_to_current[i] = node_new_comm[curr];
    }

    // 第一遍：收集边并排序去重（避免重复边和自环）
    uint32_t *temp_edges = malloc(graph->n_edges * 2 * sizeof(uint32_t));
    if (!temp_edges) {
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    
    uint32_t edge_count = 0;
    for (uint32_t u = 0; u < old_n; ++u) {
        uint32_t u_new = node_new_comm[u];
        for (uint32_t ei = graph->row_ptr[u]; ei < graph->row_ptr[u + 1]; ++ei) {
            uint32_t v = graph->col_idx[ei];
            uint32_t v_new = node_new_comm[v];
            // 跳过自环
            if (u_new != v_new) {
                temp_edges[edge_count * 2] = u_new;
                temp_edges[edge_count * 2 + 1] = v_new;
                edge_count++;
            }
        }
    }
    
    // 排序边以便去重
    edge_pair_t *edges = (edge_pair_t*)temp_edges;
    
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < edge_count; ++i) {
        // 确保边是有序的 (小节点在前)
        if (edges[i].src > edges[i].dst) {
            uint32_t tmp = edges[i].src;
            edges[i].src = edges[i].dst;
            edges[i].dst = tmp;
        }
    }
    
    // 使用qsort排序
    qsort(edges, edge_count, sizeof(edge_pair_t), cmp_edges);
    
    // 去重边
    uint32_t unique_count = 0;
    if (edge_count > 0) {
        edges[0] = edges[0];  // 保留第一条边
        unique_count = 1;
        for (uint32_t i = 1; i < edge_count; ++i) {
            if (edges[i].src != edges[unique_count-1].src || 
                edges[i].dst != edges[unique_count-1].dst) {
                edges[unique_count++] = edges[i];
            }
        }
    }
    
    // 统计每个节点的度数
    uint32_t *degree_counts = calloc(new_n, sizeof(uint32_t));
    if (!degree_counts) {
        free(temp_edges);
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    for (uint32_t i = 0; i < unique_count; ++i) {
        degree_counts[edges[i].src]++;
        degree_counts[edges[i].dst]++;
    }
    
    // 构建CSR结构
    uint32_t *new_row_ptr = malloc((new_n + 1) * sizeof(uint32_t));
    if (!new_row_ptr) {
        free(temp_edges);
        free(degree_counts);
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    
    new_row_ptr[0] = 0;
    for (uint32_t i = 0; i < new_n; ++i) {
        new_row_ptr[i + 1] = new_row_ptr[i] + degree_counts[i];
        degree_counts[i] = 0;  // 重用作为cursor
    }
    
    uint32_t new_m = new_row_ptr[new_n];
    uint32_t *new_col_idx = malloc(new_m * sizeof(uint32_t));
    if (!new_col_idx) {
        free(temp_edges);
        free(new_row_ptr);
        free(degree_counts);
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    
    // 填充邻接表
    for (uint32_t i = 0; i < unique_count; ++i) {
        uint32_t u = edges[i].src;
        uint32_t v = edges[i].dst;
        new_col_idx[new_row_ptr[u] + degree_counts[u]++] = v;
        new_col_idx[new_row_ptr[v] + degree_counts[v]++] = u;
    }
    
    free(temp_edges);
 
    // 计算新节点度数
    uint32_t *new_degree = malloc(new_n * sizeof(uint32_t));
    if (!new_degree) {
        free(new_col_idx);
        free(new_row_ptr);
        free(degree_counts);
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    for (uint32_t i = 0; i < new_n; ++i) {
        new_degree[i] = new_row_ptr[i + 1] - new_row_ptr[i];
    }

    // 初始化新社区（每个超级节点是自己的社区）
    uint32_t *new_node_comm = malloc(new_n * sizeof(uint32_t));
    if (!new_node_comm) {
        free(new_degree);
        free(new_col_idx);
        free(new_row_ptr);
        free(degree_counts);
        free(comm_map);
        free(node_new_comm);
        return -1;
    }
    for (uint32_t i = 0; i < new_n; ++i) {
        new_node_comm[i] = i;
    }

    // 释放旧图，更新为新图
    free(graph->row_ptr);
    free(graph->col_idx);
    graph->row_ptr = new_row_ptr;
    graph->col_idx = new_col_idx;
    graph->n_nodes = new_n;
    graph->n_edges = new_m;

    // 更新度数数组
    uint32_t *old_degree = *node_degree_ptr;
    *node_degree_ptr = new_degree;
    free(old_degree);

    // 更新社区数组
    uint32_t *old_comm_array = *node_comm_ptr;
    *node_comm_ptr = new_node_comm;
    free(old_comm_array);

    // 清理临时数据
    free(comm_map);
    free(node_new_comm);
    free(degree_counts);
    
    return 1;
}
 