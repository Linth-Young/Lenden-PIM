#ifndef MEMBERSHIP_IO_H
#define MEMBERSHIP_IO_H

#include <stdint.h>
#include <stdbool.h>

/*
 * 读取社区划分文件到内存（C 版本）
 *
 * 支持两种格式（逐行）：
 *   1) "idx comm"
 *   2) "comm"            // 仅一列社区ID，索引按行号自增
 *
 * 参数:
 *   path               文件路径
 *   expected_vertices  期望顶点总数（用于越界与缺失检查）
 *   out_membership     输出：长度为 expected_vertices 的数组，调用者负责 free()
 *   out_num_communities输出：社区总数（= max(comm) + 1），可为 NULL
 *
 * 返回:
 *   true  成功
 *   false 失败（并在 stderr 打印原因）
 */
bool load_membership_file(const char *path,
                          uint32_t expected_vertices,
                          uint32_t **out_membership,
                          uint32_t *out_num_communities);

#endif /* MEMBERSHIP_IO_H */