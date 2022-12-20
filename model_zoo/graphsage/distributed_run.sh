#!/bin/bash

echo "=============================================================================================================="
echo "If device is GPU"
echo "Please run the script as: "
echo "bash distributed_run.sh GPU DATA_PATH"
echo "DATA_PATH is root path containing reddit dataset 'npz' file."
echo "=============================================================================================================="
echo "If device is Ascend"
echo "Please run the script as: "
echo "bash distributed_run.sh Ascend DATA_PATH RANK_START RANK_SIZE RANK_TABLE_FILE"
echo "DATA_PATH is root path containing reddit dataset 'npz' file."
echo "RANK_START is the first Ascend device id be used"
echo "RANK_SIZE is numbers of Ascend device be used"
echo "RANK_TABLE_FILE is root path of 'rank_table_*pcs.json' file."
# for more information about 'rank_table_*pcs.json' setting see,\
# https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/parallel/train_ascend.html
echo "=============================================================================================================="

set -e
DEVICE_TARGET=$1
DATA_PATH=$2
export DEVICE_TARGET=${DEVICE_TARGET}

execute_path=$(pwd)
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")

if [ "${DEVICE_TARGET}" = "GPU" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export CUDA_NUM=8
  rm -rf device
  mkdir device
  cp -r src ./device
  cp distributed_trainval_reddit.py ./device
  cd ./device
  echo "start training"
  mpirun --allow-run-as-root -n ${CUDA_NUM} python3 ./distributed_trainval_reddit.py --data-path ${DATA_PATH}\
   --epochs 5 > train.log 2>&1 &
fi

if [ "${DEVICE_TARGET}" = "Ascend" ]; then
  RANK_TABLE_FILE=$3
  export RANK_TABLE_FILE=${RANK_TABLE_FILE}
  for((i=0;i<8;i++));
  do
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$[i+RANK_START]
    echo ${DEVICE_ID}
    rm -rf ${execute_path}/device_$RANK_ID
    mkdir ${execute_path}/device_$RANK_ID
    cd ${execute_path}/device_$RANK_ID || exit
    echo "start training"
    python3 ${self_path}/distributed_trainval_reddit.py --data-path ${DATA_PATH} --epochs 2 > train$RANK_ID.log 2>&1 &
  done
fi
