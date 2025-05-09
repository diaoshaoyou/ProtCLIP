import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoConfig, EsmConfig
from metric import f1_max, area_under_prc, spearmanr
from model.downstream_model import EsmForSequenceClassification
from trainer.downstream_trainer import DownstreamTrainer
from KG_preprocess import preprocess

class DownstreamTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.num_labels = run_config.num_labels
        self.task_model = self.build_task_model()
        self.dataset = self.build_dataset()
        self.train_args = self.build_train_args()
        self.trainer = self.build_trainer()

    def build_task_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        protein_tokenizer = AutoTokenizer.from_pretrained('/root/data/backbones/esm2_t33_650M_UR50D')

        def preprocess_function(examples):
            tokenized_examples = protein_tokenizer(examples["seq"], truncation=True,
                                                   padding="max_length",
                                                   max_length=self.run_config.max_length)
            tokenized_examples['label'] = torch.tensor(examples['label'])
            return tokenized_examples

        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.json',
            'valid': f'{self.run_config.data_path}/{self.run_config.dataset}/valid.json',
            'test': f'{self.run_config.data_path}/{self.run_config.dataset}/test.json',
        })
        dataset = dataset.map(preprocess_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=self.run_config.metric_for_best_model,
            greater_is_better=False if self.run_config.metric_for_best_model in ['mae', 'rmse'] else True,
            fp16=self.run_config.fp16,
            learning_rate=self.run_config.lr,
            push_to_hub=False,
            save_total_limit=1,
            report_to=["none"],
            deepspeed=self.run_config.deepspeed_config,
            local_rank=self.run_config.local_rank,
            gradient_checkpointing=True
        )

    def build_trainer(self):
        trainer = DownstreamTrainer(
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            protein_model_fixed=self.run_config.protein_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
        )
        return trainer

    def compute_metrics(self, eval_pred):
        raise NotImplementedError()

    def run(self):
        self.trainer.train()
        self.trainer.evaluate(self.dataset["test"], metric_key_prefix="test")


class SingleLabelSequenceClassificationTask(DownstreamTask):
    def __init__(self, run_config):
        super().__init__(run_config)

    def build_task_model(self):
        model_config = AutoConfig.from_pretrained(self.run_config.protein_model_name)
        return EsmForSequenceClassification(model_config, 
                                            self.run_config.num_labels)
        # if self.run_config.MOLE == 0:
        #     model_config = AutoConfig.from_pretrained(self.run_config.protein_model_name)
        #     return EsmForSequenceClassification(model_config, 
        #                                         self.run_config.num_labels,
        #                                         self.run_config.MOLE)
        # elif self.run_config.MOLE == 1:
        #     model_config = EsmConfig(**self.run_config.protein_model_config)
        #     return EsmForSequenceClassification(model_config, 
        #                                         self.run_config.num_labels,
        #                                         self.run_config.MOLE,
        #                                         protein_model_state=self.run_config.protein_model_state,
        #                                         lora_r=self.run_config.lora_r,
        #                                         mole_num_experts=self.run_config.mole_num_experts,
        #                                         mole_gate_mode=self.run_config.mole_gate_mode,
        #                                         lora_alpha=self.run_config.lora_alpha,
        #                                         lora_dropout=self.run_config.lora_dropout,
        #                                         lora_target_modules=self.run_config.lora_target_modules)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": (predictions == labels).mean(),
            "matthews correlation coefficient": matthews_corrcoef(labels, predictions)
        }


class MultiLabelSequenceClassificationTask(SingleLabelSequenceClassificationTask):
    def __init__(self, run_config):
        super().__init__(run_config)

    def build_trainer(self):
        def collate_fn(examples):
            labels = torch.stack([example['label'] for example in examples])
            input_ids = torch.stack([example['input_ids'] for example in examples])
            attention_mask = torch.stack([example['attention_mask'] for example in examples])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        trainer = DownstreamTrainer(
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            compute_metrics=self.compute_metrics,
            data_collator=collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            protein_model_fixed=self.run_config.protein_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
        )
        return trainer

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        return {
            "f1_max": f1_max(torch.tensor(predictions), torch.tensor(labels)),
            "auprc_micro": area_under_prc(torch.tensor(predictions).flatten(), torch.tensor(labels).long().flatten())
        }


class SequenceRegressionTask(SingleLabelSequenceClassificationTask):
    def __init__(self, run_config):
        super().__init__(run_config)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = torch.tensor(predictions).squeeze()
        labels = torch.tensor(labels).squeeze()
        tmp = ((predictions - labels) ** 2).mean().float()
        rmse = tmp.sqrt().half()
        return {
            "mae": (predictions - labels).abs().mean(),
            "rmse": rmse,
            "spearman": spearmanr(predictions, labels)
        }

class TransferTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        # if self.run_config.MOLE == 1: 
        #     self.protein_model = AutoModel.from_pretrained('/root/data/backbones/esm2_t33_650M_UR50D')
        #     MOLE_config = MOLEConfig(
        #                 task_type=TaskType.FEATURE_EXTRACTION,
        #                 inference_mode=True,
        #                 r=run_config.lora_r,
        #                 num_experts=run_config.mole_num_experts,
        #                 gate_mode=run_config.mole_gate_mode,
        #                 lora_alpha=run_config.lora_alpha,
        #                 lora_dropout=run_config.lora_dropout,
        #                 target_modules=run_config.lora_target_modules,
        #             )
        #     self.protein_model = get_peft_model(self.protein_model, MOLE_config)
        #     self.protein_model.load_state_dict(run_config.protein_model_state)

    def run(self):
        print('!!Starting KG preprocess!!')
        # if self.run_config.MOLE == 0:
        if True:
            preprocess(model_name=self.run_config.protein_model_name, batch_size=32, output_path=self.run_config.output_path)
        # elif self.run_config.MOLE == 1:
        #     preprocess(model_name=self.run_config.protein_model_name, batch_size=32, model=self.protein_model)