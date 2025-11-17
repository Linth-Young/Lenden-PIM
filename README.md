# Leiden社区检测算法 - UPMEM PIM加速版本

基于UPMEM DPU的Leiden社区检测算法实现，支持增量计算和稀疏映射优化。

## 项目结构

```
leiden/
├── host/               # Host端代码
│   ├── host.c         # 原始版本
│   ├── host_sparse.c  # 稀疏映射优化版本
│   └── host_helpers.c # 辅助函数
├── dpu/               # DPU端代码
│   ├── dpu.c          # 原始版本
│   └── dpu_sparse.c   # 稀疏映射版本
├── support/           # 头文件和工具
│   ├── host_helpers.h # Host辅助函数头文件
│   └── cyclecnt.h     # DPU性能计数
├── graph/             # 测试图数据
│   ├── amazon_sym.txt              # Amazon图 (335K节点, 925K边)
│   ├── amazon_part_leiden.txt      # Amazon初始社区分配
│   ├── web-NotreDame.edges         # Web图 (325K节点)
│   ├── web-NotreDame_communities.txt
│   ├── germany_osm_edgelist.txt    # 德国路网图 (11.5M节点, 23M边)
│   └── germany_osm_part_leiden.txt
├── results/           # 测试结果
│   ├── amazon/        # Amazon图测试结果
│   └── web/           # Web图测试结果
├── Makefile           # 原始版本编译
├── makefile_sparse    # 稀疏版本编译
└── test_optimized.sh  # 自动化测试脚本
```

## 功能特性

### 核心算法
- **Leiden社区检测**: 基于模块度优化的社区发现算法
- **增量计算**: 支持只对部分节点进行社区优化（0.1%-100%）
- **多层聚合**: 支持图收缩和多层次社区检测
- **模块度评估**: 自动计算初始和最终模块度，显示提升效果

### 性能优化
1. **多线程并行构建稀疏映射** (OpenMP)
2. **Broadcast传输共享数据** (batch参数 + 顶点列表)
3. **异步传输减少等待时间**
4. **传输数据量统计和监控**
5. **8字节对齐优化** (满足MRAM要求)

## 快速开始

### 环境要求
- UPMEM SDK (已安装DPU工具链)
- GCC with OpenMP支持

### 编译

```bash
# 编译稀疏优化版本
make -f makefile_sparse

# 编译原始版本
make
```

### 运行测试

使用自动化测试脚本（推荐）:

```bash
# Amazon图测试（默认）
./test_optimized.sh amazon

# Web图测试
./test_optimized.sh web

# 德国路网图测试
./test_optimized.sh germany
```

脚本会自动测试不同增量比例（0.1%, 1%, 10%, 20%, 30%），结果保存在 `test_results/` 目录。

### 手动运行

```bash
# 基本用法
bin/host <图文件> <DPU数量> <最大迭代次数> <输出CSV> [初始社区文件]

# 示例：Amazon图，64个DPU，1次迭代，10%增量
DELTA_PERCENT=10 bin/host \
    graph/amazon_sym.txt \
    64 \
    1 \
    output.csv \
    graph/amazon_part_leiden.txt

# 全量模式（100%节点）
DELTA_PERCENT=100 bin/host graph/amazon_sym.txt 64 1 output.csv
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `DELTA_PERCENT` | 增量计算比例 (0.1-100) | 环境变量控制 |
| `OMP_NUM_THREADS` | OpenMP线程数 | 64 |
| 图文件 | edgelist格式的图文件 | 必需 |
| DPU数量 | 使用的DPU数量 | 必需 |
| 最大迭代次数 | 每层最大迭代次数 | 1 |
| 输出CSV | 结果输出文件 | 必需 |
| 初始社区文件 | 预先分配的社区 (可选) | - |

## 输出说明

### 模块度对比
```
========================================
模块度对比 (Modularity Comparison)
========================================
初始模块度: Q₀ = 0.961461
最终模块度: Q  = 0.961684
模块度提升: ΔQ = 0.000222 ↑ (+0.02%)
========================================
```

### 性能分析
```
=== Host Profiling (seconds) ===
总运行时间: 3.018 s (DPU内核 1.234 s)
**[A]** Load edges/CSR build: 0.500000
**[B]** Initial partition & comm totals: 0.100000
**[C]** Build sparse comm maps (batch prep): 0.300000
**[D]** Transfer maps + params to DPU: 0.200000
**[E]** DPU kernel execution: 1.234000
**[F]** Fetch results from DPU: 0.050000
**[G]** Merge community updates: 0.100000
**[H]** Logging / CSV output: 0.001000
**[I]** Graph contraction & repartition: 0.400000
**[J]** Modularity evaluation: 0.033000
```

### 输出文件
- **CSV文件**: `<测试名>_opt_<增量比例>.csv` - 节点社区分配
- **日志文件**: `<测试名>_opt_<增量比例>.log` - 完整运行日志

## 图数据格式

### Edgelist格式
```
源节点 目标节点
0 1
0 2
1 2
...
```

### 初始社区文件格式
```
节点ID 社区ID
0 0
1 0
2 1
...
```

## 性能调优

### OpenMP线程数
```bash
# 根据CPU核心数调整
export OMP_NUM_THREADS=32  # 使用32个线程
```

### 批次大小
修改 `Makefile` 中的 `BATCH_SIZE`:
```c
#define BATCH_SIZE 65536  // 默认64K，可根据图规模调整
```

### DPU数量
根据可用DPU和图规模选择:
- 小图(< 1M节点): 16-32 DPU
- 中图(1M-10M节点): 32-64 DPU
- 大图(> 10M节点): 64+ DPU

## 已知限制

1. **图规模**: 受DPU MRAM限制（64MB/DPU）
2. **邻居数量**: 最大外部邻居数 `MAX_NEIGHBOR_COMMS = 1,200,000`
3. **社区数量**: 与图节点数相关
4. **边数据类型**: 当前仅支持无权图

## 测试结果

测试结果保存在 `results/` 目录:
- `amazon/`: Amazon图测试结果
- `web/`: Web图测试结果
- `germany/`: 德国路网图测试结果（如有）

每个结果包含不同增量比例下的社区分配和性能日志。

