import transformers
import wandb
import os
from model.pretrain_model import FSTForPretrain
import deepspeed
import argparse
import yaml
import json
from safetensors.torch import load_file


wandb.init(mode='disabled')
transformers.logging.set_verbosity_info()
downstream_tasks = __import__('task.downstream_task', fromlist='*')

def main():
    parser = argparse.ArgumentParser(description='FST')
    parser.add_argument('--task_name', type=str, default='esm2_t33_650M_UR50D_SubcellularLocalization')
    # path
    parser.add_argument('--data_path', type=str, default='/root/data/datasets/ProtSTData')
    parser.add_argument('--output_path', type=str, default='/root/data/FST/SubLoc/')
    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--protein_model_name', type=str, default='best_ckpt')
    parser.add_argument('--protein_model_fixed', type=bool, default=False)
    parser.add_argument('--text_model_name', type=str, default='/root/data/backbones/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--text_model_fixed', type=bool, default=True)
    # dataset
    parser.add_argument('--dataset', type=str, default='SubcellularLocalization')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_labels', type=int, default=10)
    # train
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lr_ratio', type=float, default=0.1)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # eval
    parser.add_argument('--metric_for_best_model', type=str, default='accuracy')
    # task
    parser.add_argument('--task', type=str, default='SingleLabelSequenceClassificationTask')
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

    # if args.MOLE == 0:
    if True:
        model = FSTForPretrain.from_pretrained(args.model_name)
        model.protein_model.save_pretrained(args.protein_model_name)
    # elif args.MOLE == 1:
    #     # config
    #     with open(f'{args.model_name}/config.json', 'r') as f:
    #         model_config = json.load(f)
    #     args.protein_model_config = model_config['protein_model_config']

    #     # model state dict
    #     all_state_dict = load_file(f'{args.model_name}/model.safetensors')
    #     protein_model_state = {k[len('protein_model.'): ]: v for k, v in all_state_dict.items() if k.startswith('protein_model.')}
    #     args.protein_model_state = protein_model_state
    task = getattr(downstream_tasks, args.task)(args)
    task.run()


if __name__ == '__main__':
    main()
