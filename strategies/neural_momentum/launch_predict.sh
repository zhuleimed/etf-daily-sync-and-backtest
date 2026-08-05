#!/bin/bash
# Neural Momentum 并发预测启动脚本（铁律一：bash 循环 + nohup 独立进程）
# 用法: bash launch_predict.sh [并发数]
set -e
cd "$(dirname "$0")"
PY=/home/zhulei/anaconda3/envs/zhulei_py312/bin/python
CONC=${1:-12}

ETFS="510050 510300 510500 512100 159915 588000 510180 512720 515790 516010 515050 159825 512010 515170 515220 512200"

mkdir -p ../../logs
for etf in $ETFS; do
    if [ -f "output/scores_${etf}.csv" ]; then
        echo "[$etf] 已完成，跳过"
        continue
    fi
    # 等待并发槽
    while [ $(pgrep -fc "predict_one.py --etf" 2>/dev/null || echo 0) -ge $CONC ]; do
        sleep 5
    done
    env -u KMP_AFFINITY -u OMP_NUM_THREADS nohup $PY -m strategies.neural_momentum.predict_one --etf $etf \
        > ../../logs/predict_${etf}.log 2>&1 &
    echo "[$etf] 启动 PID $!"
done

# 等待全部完成
while pgrep -f "predict_one.py --etf" > /dev/null; do sleep 10; done
echo "=== 全部完成 ==="
ls output/scores_*.csv | wc -l
