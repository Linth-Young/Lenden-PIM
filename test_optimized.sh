#!/bin/bash
# 测试优化后的稀疏版本
#
# 使用方法:
#   Amazon图测试（默认）: ./test_optimized.sh amazon
#   Web图测试:            ./test_optimized.sh web
#   德国路网图测试:       ./test_optimized.sh germany
#
# 支持的图:
#   amazon  - Amazon图 (335K nodes, 925K edges)
#   web     - Web Notre Dame图 (325K nodes)
#   germany - Germany OSM图 (11.5M nodes, 23M edges)

set -e

# 获取测试名称参数（默认为amazon）
TEST_NAME=${1:-amazon}

echo "=========================================="
echo "编译优化版本（多线程 + broadcast + 统计）"
echo "=========================================="


# 设置线程数
export OMP_NUM_THREADS=32
echo "OpenMP线程数: $OMP_NUM_THREADS"

# 清理并编译
echo "清理旧文件..."
rm -rf bin 2>/dev/null || true

echo "编译稀疏版本..."
make

echo ""
echo "=========================================="

# 根据测试名称选择图
case "$TEST_NAME" in
    germany)
        GRAPH="graph/germany_osm_edgelist.txt"
        MEMBERSHIP="graph/germany_osm_part_leiden.txt"
        echo "运行测试 - 德国路网图 (11.5M nodes)"
        echo "=========================================="
        ;;
    web)
        GRAPH="graph/web-NotreDame.edges"
        MEMBERSHIP="graph/web-NotreDame_communities.txt"
        echo "运行测试 - Web Notre Dame图 (325K nodes)"
        echo "=========================================="
        ;;
    amazon)
        GRAPH="graph/amazon_sym.txt"
        MEMBERSHIP="graph/amazon_part_leiden.txt"
        echo "运行测试 - Amazon图 (335K nodes)"
        echo "=========================================="
        ;;
    *)
        echo "错误: 未知的测试名称 '$TEST_NAME'"
        echo "支持的测试: amazon, web, germany"
        echo "使用方法: ./test_optimized.sh [amazon|web|germany]"
        exit 1
        ;;
esac

# 检查文件是否存在
if [ ! -f "$GRAPH" ]; then
    echo "错误: 图文件不存在: $GRAPH"
    echo "提示: 请确认图文件路径或使用其他测试"
    exit 1
fi

if [ ! -f "$MEMBERSHIP" ]; then
    echo "警告: Membership文件不存在: $MEMBERSHIP"
    echo "       将使用默认初始化（每个节点独立）"
    MEMBERSHIP=""
fi

NUM_DPUS=64
MAX_ITERS=1

# 测试不同增量比例
for DELTA in 0.1 1 10 20 30; do
    echo ""
    echo "=== 测试增量比例: ${DELTA}% ==="
    
    export DELTA_PERCENT=$DELTA
    OUTPUT_CSV="results/${TEST_NAME}/opt_${DELTA}.csv"
    LOG_FILE="results/${TEST_NAME}/opt_${DELTA}.log"
    
    mkdir -p results/$TEST_NAME
    
    if [ -z "$MEMBERSHIP" ]; then
        echo "运行命令: DELTA_PERCENT=$DELTA bin/host $GRAPH $NUM_DPUS $MAX_ITERS $OUTPUT_CSV"
        ./bin/host "$GRAPH" $NUM_DPUS $MAX_ITERS "$OUTPUT_CSV" 2>&1 | tee "$LOG_FILE"
    else
        echo "运行命令: DELTA_PERCENT=$DELTA bin/host $GRAPH $NUM_DPUS $MAX_ITERS $OUTPUT_CSV $MEMBERSHIP"
        ./bin/host "$GRAPH" $NUM_DPUS $MAX_ITERS "$OUTPUT_CSV" "$MEMBERSHIP" 2>&1 | tee "$LOG_FILE"
    fi
    
    echo ""
    echo "日志已保存: $LOG_FILE"
    
    # 提取关键性能指标
    echo "=== 性能摘要 ==="
    grep -E "(总运行时间|DPU内核|Build sparse|Transfer maps)" "$LOG_FILE" || true
    echo ""
done

echo ""
echo "=========================================="
echo "优化效果对比"
echo "=========================================="
echo ""
echo "关键优化:"
echo "  1. ✅ 多线程并行构建稀疏映射（OpenMP）"
echo "  2. ✅ 使用broadcast传输共享数据（batch参数+顶点列表）"
echo "  3. ✅ 异步传输减少等待时间"
echo "  4. ✅ 传输数据量统计和监控"
echo ""
echo "✓ 测试完成！结果保存在 results/${TEST_NAME}/opt_*.csv 和 results/${TEST_NAME}/opt_*.log"
