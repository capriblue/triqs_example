#!/bin/bash
#SBATCH -N 1
#SBATCH -p FOMP
#SBATCH --exclude=orchid1,orchid2


#!/bin/bash

# 開始時間を取得
start_time=$(date +%s)

# 測定したいコードを実行
# ここにコードを追加してください
mpirun -np 16 python Uneq0.py

# 終了時間を取得
end_time=$(date +%s)

# 実行時間を計算
execution_time=$((end_time - start_time))

# 結果を出力
echo "実行時間: ${execution_time}秒"
