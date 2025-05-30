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
    parser.add_argument('--data_path', type=str, default='/root/DATA/datasets')
    parser.add_argument('--output_path', type=str, default='/root/DATA/downstream/SubLoc')
    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--protein_model_name', type=str, default='best_ckpt')
    parser.add_argument('--protein_model_fixed', type=bool, default=False)
    parser.add_argument('--text_model_name', type=str, default='/root/DATA/backbones/BiomedNLP-PubMedBERT-base-uncased-abstract')
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
    # RANK
    parser.add_argument('--local_rank', type=int, default=0, help='machine local_rank')
    # deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    model = FSTForPretrain.from_pretrained(args.model_name)
    model.protein_model.save_pretrained(args.protein_model_name)
    
    task = getattr(downstream_tasks, args.task)(args)
    task.run()


if __name__ == '__main__':
    main()
