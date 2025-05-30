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
torchrun $ddp_options /root/CODE/protclip/pretrain.py \
--deepspeed --deepspeed_config /root/CODE/protclip/zero1.json \
--batch_size 64 \
--output_path /root/DATA/checkpoint