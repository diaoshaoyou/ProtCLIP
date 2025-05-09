import hydra
import transformers
import wandb
import os
from model.pretrain_model import FSTForPretrain
import deepspeed
import argparse
import yaml

wandb.init(mode='disabled')

transformers.logging.set_verbosity_info()
downstream_tasks = __import__('task.downstream_task', fromlist='*')
pretrain_tasks = __import__('task.pretrain_task', fromlist='*')

def main():
    parser = argparse.ArgumentParser(description='FST')
    parser.add_argument('--task_name', type=str, default='esm2_t33_650M_UR50D_ProtST')
    # path
    parser.add_argument('--data_path', type=str, default='/root/data/datasets/ProtSTData')
    parser.add_argument('--output_path', type=str, default='/root/data/FST/')
    # model
    parser.add_argument('--protein_model_name', type=str, default='/root/data/backbones/esm2_t33_650M_UR50D')
    parser.add_argument('--protein_model_fixed', type=bool, default=False)
    parser.add_argument('--text_model_name', type=str, default='/root/data/backbones/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--text_model_fixed', type=bool, default=True)
    parser.add_argument('--projection_dim', type=int, default=512)
    parser.add_argument('--fusion_num_heads', type=int, default=8)
    parser.add_argument('--fusion_num_layers', type=int, default=1)
    parser.add_argument('--fusion_batch_norm', type=bool, default=True)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--protein_mask_range', type=int, default=10)
    parser.add_argument('--proto_dim', type=int, default=512)
    parser.add_argument('--proto_num', type=int, default=512)
    parser.add_argument('--local_pool', type=str, default='avg')
    parser.add_argument('--fragment_range', type=int, default=60)
    # dataset
    parser.add_argument('--dataset', type=str, default='ProtDescribe')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--text_attributes', nargs='+', default=['PROTEIN NAME', 'FUNCTION', 'SUBCELLULAR LOCATION', 'SIMILARITY'])
    parser.add_argument('--noise_lambda', type=int, default=3)
    # train
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_ratio', type=float, default=0.1)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=35)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    # eval
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    # task
    parser.add_argument('--task', type=str, default='FSTPretrainTask')
    # loss ablation
    parser.add_argument('--mmp', type=int, default=0)
    parser.add_argument('--t2p_mlm', type=int, default=1)
    parser.add_argument('--local_contrast', type=int, default=1)
    # MOLE
    # parser.add_argument('--MOLE', type=int, default=0)
    # parser.add_argument('--lora_r', type=int, default=32)
    # parser.add_argument('--lora_alpha', type=int, default=32)
    # parser.add_argument('--lora_dropout', type=float, default=0.1)
    # parser.add_argument('--lora_target_modules', nargs='+', default=['query', 'key', 'value', 'dense'])
    # parser.add_argument('--mole_num_experts', type=int, default=6)
    # parser.add_argument('--mole_gate_mode', type=str, default='top2_gate')
    # RANK
    parser.add_argument('--local_rank', type=int, default=0, help='machine local_rank')
    # deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    task = getattr(pretrain_tasks, args.task)(args)
    task.run()


if __name__ == '__main__':
    main()
