export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL


echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l) # use all_available gpus
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "worker-gpu: $(nvidia-smi -L | wc -l)"

echo "================================ddp_options===================================="
ddp_options="--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$n_gpu --nnodes=$WORLD_SIZE --node_rank=$RANK"
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="


# run
torchrun $ddp_options /root/code/protein/FST/downstream.py \
--deepspeed --deepspeed_config /root/code/protein/FST/zero1.json \
--output_path /root/data/FST/GOMF/ff_mlm0.7_5 \
--model_name /root/data/FST/mlm0.7/checkpoint-5280 \
--task MultiLabelSequenceClassificationTask \
--dataset GeneOntology_MF \
--num_labels 489 \
--metric_for_best_model f1_max 

# --task SingleLabelSequenceClassificationTask \
# --dataset  SubcellularLocalization  \
# --num_labels  10  \
# --metric_for_best_model accuracy \

# --task SequenceRegressionTask \
# --dataset Thermostability \
# --num_labels 1 \
# --metric_for_best_model mae















