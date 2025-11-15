#include "membership_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* C 实现：从文件读取每个顶点的社区ID并做一致性校验 */
bool load_membership_file(const char *path,
                          uint32_t expected_vertices,
                          uint32_t **out_membership,
                          uint32_t *out_num_communities)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("Failed to open membership file");
        return false;
    }

    uint32_t *membership = (uint32_t *)malloc(expected_vertices * sizeof(uint32_t));
    if (!membership) {
        fprintf(stderr, "Failed to allocate membership array\n");
        fclose(fp);
        return false;
    }
    /* 用 UINT32_MAX 标记“尚未赋值” */
    for (uint32_t i = 0; i < expected_vertices; ++i) {
        membership[i] = UINT32_MAX;
    }

    uint64_t assigned = 0;           /* 成功赋值的条目数 */
    uint32_t max_comm = 0;           /* 最大社区ID */
    uint32_t next_vertex = 0;        /* 仅一列格式时的隐式索引 */

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        /* 跳过空行/注释行 */
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }

        uint32_t idx  = UINT32_MAX;
        uint32_t comm = UINT32_MAX;
        /* 支持两种输入格式：
           1) "idx comm" → consumed == 2
           2) "comm"     → consumed == 1，此时 idx 使用 next_vertex 自增 */
        int consumed = sscanf(line, "%u %u", &idx, &comm);
        if (consumed == 1) {
            /* 仅有一个数字，则视为社区ID，索引隐式递增 */
            comm = idx;
            idx  = next_vertex;
        } else if (consumed != 2) {
            fprintf(stderr, "Malformed membership line: %s", line);
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

        /* 若当前行采用隐式索引，推进 next_vertex */
        if (consumed == 1 && idx == next_vertex) {
            next_vertex++;
        }
    }
    fclose(fp);

    if (assigned == 0) {
        fprintf(stderr, "Membership file appears to be empty\n");
        free(membership);
        return false;
    }

    /* 校验是否每个顶点都已赋值 */
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