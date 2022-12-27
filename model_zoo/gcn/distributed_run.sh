#!/bin/bash

echo "=============================================================================================================="
echo "If device is GPU"
echo "Please run the script as: "
echo "bash model_zoo/gcn/distributed_run.sh GPU DATA_PATH"
echo "DATA_PATH is root path containing reddit dataset 'npz' file."
echo "=============================================================================================================="
echo "If device is Ascend(default ranksize=8)"
echo "Please run the script as: "
echo "bash model_zoo/gcn/distributed_run.sh DEVICE DATA_PATH"
echo "DATA_PATH is root path containing reddit dataset 'npz' file."
echo "example: bash model_zoo/gcn/distributed_run.sh Ascend /home/.../data"
echo "=============================================================================================================="

set -e
DEVICE_TARGET=$1
DATA_PATH=$2
export DEVICE_TARGET=${DEVICE_TARGET}

if [ "${DEVICE_TARGET}" = "GPU" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_NUM=8
    rm -rf device
    mkdir device
    cp -r model_zoo/gcn/src ./device
    cp model_zoo/gcn/distributed_trainval_reddit.py ./device
    cd ./device
    echo "start training"
    mpirun --allow-run-as-root -n ${CUDA_NUM} python3 ./distributed_trainval_reddit.py --data_path ${DATA_PATH}\
     --epochs 5 > train.log 2>&1 &
fi

if [ "${DEVICE_TARGET}" = "Ascend" ]; then
    export RANK_SIZE=2
    export HCCL_CONNECT_TIMEOUT=6000
    test_dist_2pcs()
    {
        # export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
        export RANK_TABLE_FILE=/user/config/nbstart_hccl.json
        export RANK_SIZE=2
    }

    test_dist_${RANK_SIZE}pcs

    for((i=0;i<${RANK_SIZE};i++))
    do
        rm -rf device$i
        mkdir device$i
        cp -r model_zoo/gcn/src ./device$i
        cp model_zoo/gcn/distributed_trainval_reddit.py ./device$i
        cd ./device$i
        export DEVICE_ID=$i
        export RANK_ID=$i
        echo "start training for device $i"
        env > env$i.log
        python ./distributed_trainval_reddit.py --data_path ${DATA_PATH} --epochs 5 > train.log$i 2>&1 &
        cd ../
    done
fi
echo "The program launch succeed, the log is under device0/train.log0."