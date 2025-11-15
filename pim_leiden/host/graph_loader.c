#include "graph_loader.h"

bool load_graph(const char *filename, graph_t *graph) {
    memset(graph, 0, sizeof(*graph));
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Unable to open graph file");
        return false;
    }

    uint32_t hint_nodes = 0, hint_edges = 0;
    if (fscanf(fp, "%u %u", &hint_nodes, &hint_edges) != 2) {
        fprintf(stderr, "Graph file format error (expected N M on first line)\n");
        fclose(fp);
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
        if (read == EOF) break;
        if (read != 2) {
            fprintf(stderr, "Graph file format error around edge %u\n", edge_index);
            free(src); free(dst); fclose(fp);
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
                return false;
            }
            src = new_src;
            dst = new_dst;
            capacity_edges = new_capacity;
        }
        src[edge_index] = u;
        dst[edge_index] = v;
        if (u > max_node_id) max_node_id = u;
        if (v > max_node_id) max_node_id = v;
        ++edge_index;
    }
    fclose(fp);

    if (edge_index == 0) {
        fprintf(stderr, "The graph contains no edges.\n");
        free(src); free(dst);
        return false;
    }


    size_t pair_count = (size_t)edge_index;
    uint32_t *all_ids = malloc(pair_count * 2U * sizeof(uint32_t));
    if (!all_ids) {
        fprintf(stderr, "Memory allocation failed while staging node IDs\n");
        free(src); free(dst);
        return false;
    }

    for (size_t i = 0; i < pair_count; ++i) {
        all_ids[2 * i]     = src[i];
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
        free(all_ids); free(src); free(dst);
        return false;
    }

    uint32_t *new_to_old = malloc(unique_count * sizeof(uint32_t));
    if (!new_to_old) {
        fprintf(stderr, "Failed to allocate ID remap array.\n");
        free(all_ids); free(src); free(dst);
        return false;
    }
    memcpy(new_to_old, all_ids, unique_count * sizeof(uint32_t));
    uint32_t max_old_id = new_to_old[unique_count - 1];
    uint32_t *old_to_new = malloc(((size_t)max_old_id + 1U) * sizeof(uint32_t));
    if (!old_to_new) {
        fprintf(stderr, "Failed to allocate reverse ID mapping.\n");
        free(new_to_old); free(all_ids); free(src); free(dst);
        return false;
    }
    for (uint32_t i = 0; i <= max_old_id; ++i) old_to_new[i] = UINT32_MAX;
    for (uint32_t i = 0; i < unique_count; ++i) old_to_new[new_to_old[i]] = i;

    for (uint32_t i = 0; i < edge_index; ++i) {
        uint32_t u = src[i];
        uint32_t v = dst[i];
        uint32_t nu = (u <= max_old_id) ? old_to_new[u] : UINT32_MAX;
        uint32_t nv = (v <= max_old_id) ? old_to_new[v] : UINT32_MAX;
        if (nu == UINT32_MAX || nv == UINT32_MAX) {
            fprintf(stderr, "Edge references unknown node (%u,%u).\n", u, v);
            free(old_to_new); free(new_to_old); free(all_ids); free(src); free(dst);
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
        free(new_to_old); free(src); free(dst);
        return false;
    }

    for (uint32_t e = 0; e < edge_index; ++e) {
        uint32_t u = src[e];
        graph->row_ptr[u + 1]++;
    }
    for (uint32_t i = 0; i < unique_count; ++i)
        graph->row_ptr[i + 1] += graph->row_ptr[i];

    uint32_t *cursor = malloc(unique_count * sizeof(uint32_t));
    if (!cursor) {
        fprintf(stderr, "Failed to allocate cursor buffer\n");
        free_graph(graph);
        free(new_to_old); free(src); free(dst);
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

    graph->n_nodes    = unique_count;
    graph->n_edges    = edge_index;
    graph->new_to_old = new_to_old;
    return true;
}